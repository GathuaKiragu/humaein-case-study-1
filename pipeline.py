import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import sys

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define constants
TODAY = datetime(2025, 7, 30).date()  # Given today's date
RETRYABLE_REASONS = {"missing modifier", "incorrect npi", "prior auth required"}
NON_RETRYABLE_REASONS = {"authorization expired", "incorrect provider type"}
UNIFIED_SCHEMA = ["claim_id", "patient_id", "procedure_code", "denial_reason", "status", "submitted_at", "source_system"]



def ingest_alpha(file_path):
    """Ingest and normalize data from the alpha EMR source (CSV)."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Ingested {len(df)} records from alpha source.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame(columns=UNIFIED_SCHEMA) # Return empty DataFrame

    normalized_records = []
    for _, row in df.iterrows():
        try:
            # Split the combined 'submitted_at_status' field
            date_str, status = row['submitted_at_status'].split(',')
            # Create a normalized record dictionary
            record = {
                "claim_id": str(row['claim_id']).strip(),
                "patient_id": str(row['patient_id']).strip() if pd.notna(row['patient_id']) else None,
                "procedure_code": str(row['procedure_code']).strip(),
                "denial_reason": str(row['dental_reason']).strip().lower() if pd.notna(row['dental_reason']) and str(row['dental_reason']).strip().lower() != 'none' else None,
                "status": status.strip().lower(),
                "submitted_at": pd.to_datetime(date_str).date(), # Normalize to date object
                "source_system": "alpha"
            }
            normalized_records.append(record)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping malformed row in alpha source: {row}. Error: {e}")
            continue # Skip this row and move to the next

    normalized_df = pd.DataFrame(normalized_records)
    logger.info(f"Successfully normalized {len(normalized_df)} records from alpha.")
    return normalized_df


def mock_llm_classifier(denial_reason):
    """Mock function to simulate an LLM classifying an ambiguous denial reason."""
    # This is a simple hardcoded lookup. A real implementation would be more complex.
    ambiguous_mapping = {
        "form incomplete": "retryable",
        "incorrect procedure": "retryable",
        "not billable": "non-retryable",
        None: "non-retryable"  # null reasons are not retryable
    }
    # Return the classification, defaulting to 'non-retryable' if reason is unknown
    return ambiguous_mapping.get(denial_reason, "non-retryable")


def ingest_beta(file_path):
    """Ingest and normalize data from the beta EMR source (JSON with inconsistent keys)."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Ingested {len(data)} records from beta source.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame(columns=UNIFIED_SCHEMA)

    normalized_records = []
    for record in data:
        try:
            # Map inconsistent keys to our schema. This is the crucial part.
            # This logic might need to be adjusted based on the actual variety of keys.
            claim_id = record.get('id') or record.get('identifier', '')
            patient_id = record.get('author_id') or record.get('patient', '')
            procedure_code = record.get('code') or record.get('procedure_cd', '')
            denial_reason = record.get('denied_because') or record.get('reason')
            date_status = record.get('date_status') or record.get('submitted', '')

            # Clean and standardize the data
            denial_reason = str(denial_reason).strip().lower() if denial_reason is not None and str(denial_reason).strip().lower() != 'none' else None
            # Split the combined 'date_status' field (assuming format 'YYYY-MM-DD-status')
            date_str, status = date_status.split('-', 2)[:2], date_status.split('-', 2)[-1]
            date_str = '-'.join(date_str) # Rejoin the date part

            normalized_record = {
                "claim_id": str(claim_id).strip(),
                "patient_id": str(patient_id).strip() if patient_id else None,
                "procedure_code": str(procedure_code).strip(),
                "denial_reason": denial_reason,
                "status": status.strip().lower(),
                "submitted_at": pd.to_datetime(date_str).date(),
                "source_system": "beta"
            }
            normalized_records.append(normalized_record)
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Skipping malformed record in beta source: {record}. Error: {e}")
            continue

    normalized_df = pd.DataFrame(normalized_records)
    logger.info(f"Successfully normalized {len(normalized_df)} records from beta.")
    return normalized_df


def is_eligible_for_resubmission(record):
    """Determine if a single claim record is eligible for resubmission."""
    # Rule 1: Status must be 'denied'
    if record['status'] != 'denied':
        return False, "Status not denied"

    # Rule 2: Patient ID must not be null
    if not record['patient_id']:
        return False, "Null patient_id"

    # Rule 3: Claim must be submitted more than 7 days ago
    if (TODAY - record['submitted_at']) <= timedelta(days=7):
        return False, "Submitted within last 7 days"

    # Rule 4: Check denial reason
    reason = record['denial_reason']
    if reason in RETRYABLE_REASONS:
        return True, reason
    elif reason in NON_RETRYABLE_REASONS:
        return False, f"Non-retryable reason: {reason}"
    else:
        # Handle ambiguous reasons with a mock classifier
        classification = mock_llm_classifier(reason)
        if classification == "retryable":
            return True, f"Ambiguous reason classified as retryable: {reason}"
        else:
            return False, f"Ambiguous reason classified as non-retryable: {reason}"


def mock_llm_classifier(denial_reason):
    """Mock function to simulate an LLM classifying an ambiguous denial reason."""
    # This is a simple hardcoded lookup. A real implementation would be more complex.
    ambiguous_mapping = {
        "form incomplete": "retryable",
        "incorrect procedure": "retryable",
        "not billable": "non-retryable",
        None: "non-retryable"  # null reasons are not retryable
    }
    # Return the classification, defaulting to 'non-retryable' if reason is unknown
    return ambiguous_mapping.get(denial_reason, "non-retryable")

def main():
    """Main function to run the data pipeline."""
    logger.info("Starting claim resubmission pipeline...")

    # Step 1: Ingest and Normalize data from all sources
    df_alpha = ingest_alpha('emr_alpha.csv')
    df_beta = ingest_beta('emr_beta.json')

    # Combine DataFrames from all sources
    combined_df = pd.concat([df_alpha, df_beta], ignore_index=True)
    logger.info(f"Total records after ingestion and normalization: {len(combined_df)}")

    if combined_df.empty:
        logger.warning("No data was successfully ingested. Exiting.")
        sys.exit(1)

    # Step 2: Apply eligibility logic
    results = []
    resubmission_candidates = []
    exclusion_counts = {} # To track why claims are excluded

    for _, claim in combined_df.iterrows():
        is_eligible, reason = is_eligible_for_resubmission(claim)
        result = {
            "claim_id": claim['claim_id'],
            "eligible": is_eligible,
            "reason": reason
        }
        results.append(result)

        if is_eligible:
            # Create the output format for eligible claims
            candidate = {
                "claim_id": claim['claim_id'],
                "resubmission_reason": reason,
                "source_system": claim['source_system'],
                "recommended_changes": f"Review claim {claim['claim_id']} for potential error and resubmit." # Simple recommendation
            }
            resubmission_candidates.append(candidate)
        else:
            # Count exclusion reasons
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    # Step 3: Produce Output and Logging
    # Save the resubmission candidates to a JSON file
    with open('resubmission_candidates.json', 'w') as f:
        json.dump(resubmission_candidates, f, indent=4)
    logger.info(f"Saved {len(resubmission_candidates)} resubmission candidates to 'resubmission_candidates.json'.")

    # Print final metrics
    total_claims = len(combined_df)
    alpha_claims = len(df_alpha)
    beta_claims = len(df_beta)
    flagged_claims = len(resubmission_candidates)

    print("\n--- PIPELINE METRICS ---")
    print(f"Total claims processed: {total_claims}")
    print(f"  From source 'alpha': {alpha_claims}")
    print(f"  From source 'beta': {beta_claims}")
    print(f"Claims flagged for resubmission: {flagged_claims}")
    print("\nClaims excluded for the following reasons:")
    for reason, count in exclusion_counts.items():
        print(f"  {reason}: {count}")

    logger.info("Pipeline finished successfully.")

# This ensures the main function runs only when the script is executed directly
if __name__ == "__main__":
    main()