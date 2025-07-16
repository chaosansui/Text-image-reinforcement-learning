# read_sql.py

import pandas as pd
import logging

# Import configurations
from config import DB_CONFIG, OUTPUT_FILES

# Import utility functions and quality logic
from utils.db_utils import fetch_data_from_db
# from minio_utils import initialize_minio_client, get_minio_client, read_image_from_minio # MinIO currently excluded
from quality_logic import RuleEngine, process_data, split_data, save_to_finetune, save_to_rl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_pipeline(): # Renamed the main function for clarity within the file
    # Initialize rule engine
    rule_engine = RuleEngine()

    try:
        # 1. Fetch data from the database
        logger.info("Starting to fetch data from database...")
        df = fetch_data_from_db(**DB_CONFIG)

        if df.empty:
            logger.warning("No data retrieved. Please check database connection and query conditions.")
            return

        # 2. Currently skipping MinIO image reading, as per your request
        logger.info("Current stage skips reading image data from MinIO.")
        # Ensure these columns exist even if not populated, for consistency with quality_logic
        df['image_data_from_minio'] = None
        df['image_read_success'] = False


        # 3. Process data and generate new rules
        logger.info("Starting to process data and update rules...")
        new_rules = process_data(df, rule_engine)
        rule_engine.update_rules(new_rules)

        # 4. Split data based on quality
        logger.info("Starting data splitting...")
        high_quality, mid_quality, low_quality = split_data(df, rule_engine)

        # 5. Save results to CSV files
        logger.info("Saving result files...")
        if not high_quality.empty:
            save_to_finetune(high_quality, OUTPUT_FILES["high_quality"])
        if not mid_quality.empty:
            save_to_rl(mid_quality, OUTPUT_FILES["mid_quality"])

        # 6. Output statistics
        logger.info("\n" + "="*50)
        logger.info(f"Data splitting complete - Total: {len(df)} records")
        logger.info(f"High-quality samples: {len(high_quality)} records ({len(high_quality)/len(df):.1%})")
        logger.info(f"Mid-quality samples: {len(mid_quality)} records ({len(mid_quality)/len(df):.1%})")
        logger.info(f"Low-quality samples: {len(low_quality)} records ({len(low_quality)/len(df):.1%})")

        # Debug output example data
        if not high_quality.empty:
            sample = high_quality.iloc[0]
            logger.info("\nHigh-quality sample example:")
            logger.info(f"Prompt: {sample['prompt'][:100]}...")
            logger.info(f"Result: {str(sample['result_info'])[:100]}...")
            logger.info(f"Score: {sample.get('score', 'N/A')}")


        logger.info("\nCurrent Rule Parameters:")
        for k, v in rule_engine.current_rules.items():
            logger.info(f"{k}: {v}")

    except Exception as e:
        logger.error(f"Main process execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_data_pipeline()