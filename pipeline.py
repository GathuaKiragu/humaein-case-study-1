from claim_processor import ClaimProcessor

def main():
    processor = ClaimProcessor()
    processor.process_files('emr_alpha.csv', 'emr_beta.json')
    
    # Save results
    processor.save_results()
    processor.save_rejection_log()
    
    # Print metrics
    metrics = processor.get_metrics()
    print("\n--- PIPELINE METRICS ---")
    print(f"Total claims processed: {metrics['total_claims_processed']}")
    print(f"  From source 'alpha': {metrics['claims_from_alpha']}")
    print(f"  From source 'beta': {metrics['claims_from_beta']}")
    print(f"Claims flagged for resubmission: {metrics['claims_flagged']}")
    print(f"Failed records: {metrics['failed_records_count']}")
    print("\nClaims excluded for the following reasons:")
    for reason, count in metrics['exclusion_counts'].items():
        print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()