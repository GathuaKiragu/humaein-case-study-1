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
        # ... (Copy the entire ingest_alpha function from pipeline.py here, replacing `logger` with `self.logger`) ...
        # Remember to update the return statement to use self.UNIFIED_SCHEMA
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Ingested {len(df)} records from alpha source.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return pd.DataFrame(columns=self.UNIFIED_SCHEMA)

        normalized_records = []
        for _, row in df.iterrows():
            try:
                # ... (same logic as before) ...
                record = {
                    # ... (same as before) ...
                    "source_system": "alpha"
                }
                normalized_records.append(record)
            except (ValueError, KeyError) as e:
                error_msg = f"Skipping malformed row in alpha source: {row.to_dict()}. Error: {e}"
                self.logger.warning(error_msg)
                self.failed_records.append({"source": "alpha", "record": row.to_dict(), "error": error_msg})
                continue

        normalized_df = pd.DataFrame(normalized_records)
        self.logger.info(f"Successfully normalized {len(normalized_df)} records from alpha.")
        return normalized_df

    def ingest_beta(self, file_path: str) -> pd.DataFrame:
        """Ingest and normalize data from the beta EMR source (JSON with inconsistent keys)."""
        # ... (Copy the entire ingest_beta function from pipeline.py here, making similar changes) ...
        # Remember to add failed records to self.failed_records in the except block

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
        # ... (Copy the is_eligible_for_resubmission function, replacing the call to mock_llm_classifier with self.mock_llm_classifier) ...

    def process_files(self, alpha_path: str, beta_path: str) -> None:
        """Main method to process data from both sources."""
        self.logger.info("Starting claim resubmission pipeline...")

        # Ingest and normalize data
        df_alpha = self.ingest_alpha(alpha_path)
        df_beta = self.ingest_beta(beta_path)

        # Combine data
        self.combined_df = pd.concat([df_alpha, df_beta], ignore_index=True)
        self.logger.info(f"Total records after ingestion and normalization: {len(self.combined_df)}")

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