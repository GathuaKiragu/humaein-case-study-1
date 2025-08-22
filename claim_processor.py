import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Tuple
import sys

class ClaimProcessor:
    """A class to process claims from multiple EMR sources and determine resubmission eligibility."""

    # Define constants as class attributes
    TODAY = datetime(2025, 7, 30).date()
    RETRYABLE_REASONS = {"missing modifier", "incorrect npi", "prior auth required"}
    NON_RETRYABLE_REASONS = {"authorization expired", "incorrect provider type"}
    UNIFIED_SCHEMA = ["claim_id", "patient_id", "procedure_code", "denial_reason", "status", "submitted_at", "source_system"]

    def __init__(self):
        self.combined_df = pd.DataFrame(columns=self.UNIFIED_SCHEMA)
        self.resubmission_candidates = []
        self.exclusion_counts = {}
        self.failed_records = []  # To track malformed data for rejection log
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def ingest_alpha(self, file_path: str) -> pd.DataFrame:
        """Ingest and normalize data from the alpha EMR source (CSV)."""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Ingested {len(df)} records from alpha source.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return pd.DataFrame(columns=self.UNIFIED_SCHEMA)

        normalized_records = []
        for _, row in df.iterrows():
            try:
                # Check if submitted_at_status exists and is not null
                if pd.isna(row.get('submitted_at_status')):
                    raise ValueError("submitted_at_status is null or missing")
                
                # Split the combined 'submitted_at_status' field
                date_status = str(row['submitted_at_status']).strip()
                if ',' not in date_status:
                    raise ValueError(f"submitted_at_status missing comma separator: {date_status}")
                    
                date_str, status = date_status.split(',', 1)  # Split on first comma only
                
                # Create a normalized record dictionary
                record = {
                    "claim_id": str(row['claim_id']).strip() if pd.notna(row.get('claim_id')) else None,
                    "patient_id": str(row['patient_id']).strip() if pd.notna(row.get('patient_id')) else None,
                    "procedure_code": str(row['procedure_code']).strip() if pd.notna(row.get('procedure_code')) else None,
                    "denial_reason": str(row['dental_reason']).strip().lower() if pd.notna(row.get('dental_reason')) and str(row['dental_reason']).strip().lower() != 'none' else None,
                    "status": status.strip().lower(),
                    "submitted_at": pd.to_datetime(date_str.strip()).date(),
                    "source_system": "alpha"
                }
                normalized_records.append(record)
            except (ValueError, KeyError, TypeError) as e:
                error_msg = f"Skipping malformed row in alpha source: {dict(row)}. Error: {e}"
                self.logger.warning(error_msg)
                self.failed_records.append({"source": "alpha", "record": dict(row), "error": str(e)})
                continue

        normalized_df = pd.DataFrame(normalized_records)
        self.logger.info(f"Successfully normalized {len(normalized_df)} records from alpha.")
        return normalized_df


    def ingest_beta(self, file_path: str) -> pd.DataFrame:
        """Ingest and normalize data from the beta EMR source (JSON with inconsistent keys)."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Ingested {len(data)} records from beta source.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return pd.DataFrame(columns=self.UNIFIED_SCHEMA)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {file_path}: {e}")
            self.failed_records.append({"source": "beta", "record": "FILE LEVEL ERROR", "error": f"Invalid JSON: {e}"})
            return pd.DataFrame(columns=self.UNIFIED_SCHEMA)

        normalized_records = []
        for record in data:
            try:
                # Map inconsistent keys to our schema
                claim_id = record.get('id') or record.get('identifier', '')
                patient_id = record.get('author_id') or record.get('patient', '')
                procedure_code = record.get('code') or record.get('procedure_cd', '')
                denial_reason = record.get('denied_because') or record.get('reason')
                date_status = record.get('date_status') or record.get('submitted', '')

                # Clean and standardize the data
                denial_reason = str(denial_reason).strip().lower() if denial_reason is not None and str(denial_reason).strip().lower() != 'none' else None
                
                # Split the combined 'date_status' field - FIXED INDENTATION
                if '-' in date_status:
                    parts = date_status.split('-')
                    # Assume the date is in YYYY-MM-DD format (first 3 parts)
                    if len(parts) >= 3:
                        date_str = '-'.join(parts[:3])
                        status = '-'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                    else:
                        raise ValueError(f"Invalid date_status format: {date_status}")
                else:
                    raise ValueError(f"Unsupported date_status format: {date_status}")

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
                error_msg = f"Skipping malformed record in beta source: {record}. Error: {e}"
                self.logger.warning(error_msg)
                self.failed_records.append({"source": "beta", "record": record, "error": error_msg})
                continue

        normalized_df = pd.DataFrame(normalized_records)
        self.logger.info(f"Successfully normalized {len(normalized_df)} records from beta.")
        return normalized_df


    def mock_llm_classifier(self, denial_reason: str) -> str:
        """Mock function to simulate an LLM classifying an ambiguous denial reason."""
        # Enhance the mock classifier
        ambiguous_mapping = {
            "form incomplete": "retryable",
            "incorrect procedure": "retryable",
            "not billable": "non-retryable",
            "patient ineligible": "non-retryable",
            "benefits exhausted": "non-retryable",
            "duplicate claim": "non-retryable",
            None: "non-retryable"
        }
        # Simulate some basic text pattern matching a more advanced system might use
        if denial_reason:
            if "incomplete" in denial_reason:
                return "retryable"
            if "incorrect" in denial_reason:
                return "retryable" # Let's be optimistic about incorrect data being fixable
            if "expired" in denial_reason:
                return "non-retryable"
        return ambiguous_mapping.get(denial_reason, "non-retryable")

    def is_eligible_for_resubmission(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a single claim record is eligible for resubmission."""
        # Rule 1: Status must be 'denied'
        if record.get('status') != 'denied':
            return False, "Status not denied"

        # Rule 2: Patient ID must not be null
        if not record.get('patient_id'):
            return False, "Null patient_id"

        # Rule 3: Claim must be submitted more than 7 days ago
        submitted_date = record.get('submitted_at')
        if not submitted_date:
            return False, "Missing submission date"
        
        if (self.TODAY - submitted_date) <= timedelta(days=7):
            return False, "Submitted within last 7 days"

        # Rule 4: Check denial reason
        reason = record.get('denial_reason')
        if reason in self.RETRYABLE_REASONS:
            return True, reason
        elif reason in self.NON_RETRYABLE_REASONS:
            return False, f"Non-retryable reason: {reason}"
        else:
            # Handle ambiguous reasons with a mock classifier
            classification = self.mock_llm_classifier(reason)
            if classification == "retryable":
                return True, f"Ambiguous reason classified as retryable: {reason}"
            else:
                return False, f"Ambiguous reason classified as non-retryable: {reason}"

    def process_files(self, alpha_path: str, beta_path: str) -> None:
        """Main method to process data from both sources."""
        self.logger.info("Starting claim resubmission pipeline...")

        # Ingest and normalize data
        df_alpha = self.ingest_alpha(alpha_path)
        df_beta = self.ingest_beta(beta_path)
        
        # Debug logging to check what's returned
        self.logger.debug(f"Alpha DF type: {type(df_alpha)}, shape: {df_alpha.shape if hasattr(df_alpha, 'shape') else 'N/A'}")
        self.logger.debug(f"Beta DF type: {type(df_beta)}, shape: {df_beta.shape if hasattr(df_beta, 'shape') else 'N/A'}")

        # Combine data
        self.combined_df = pd.concat([df_alpha, df_beta], ignore_index=True)
        self.logger.info(f"Total records after ingestion and normalization: {len(self.combined_df)}")

        if self.combined_df.empty:
            self.logger.warning("No data available for processing after ingestion")
            return

        # Apply eligibility logic
        for _, claim in self.combined_df.iterrows():
            is_eligible, reason = self.is_eligible_for_resubmission(claim)
            if is_eligible:
                candidate = {
                    "claim_id": claim['claim_id'],
                    "resubmission_reason": reason,
                    "source_system": claim['source_system'],
                    "recommended_changes": f"Review claim {claim['claim_id']} based on reason '{reason}' and resubmit."
                }
                self.resubmission_candidates.append(candidate)
            else:
                self.exclusion_counts[reason] = self.exclusion_counts.get(reason, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Return processing metrics."""
        alpha_count = len(self.combined_df[self.combined_df['source_system'] == 'alpha'])
        beta_count = len(self.combined_df[self.combined_df['source_system'] == 'beta'])
        
        return {
            "total_claims_processed": len(self.combined_df),
            "claims_from_alpha": alpha_count,
            "claims_from_beta": beta_count,
            "claims_flagged": len(self.resubmission_candidates),
            "exclusion_counts": self.exclusion_counts,
            "failed_records_count": len(self.failed_records)
        }

    def save_results(self, output_path: str = 'resubmission_candidates.json') -> None:
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.resubmission_candidates, f, indent=4)
        self.logger.info(f"Saved {len(self.resubmission_candidates)} candidates to {output_path}")

    def save_rejection_log(self, log_path: str = 'rejected_records.log') -> None:
        """Save information about failed records to a log file."""
        with open(log_path, 'w') as f:
            for record in self.failed_records:
                f.write(f"{json.dumps(record)}\n")
        self.logger.info(f"Saved {len(self.failed_records)} rejected records to {log_path}")