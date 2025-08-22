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