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