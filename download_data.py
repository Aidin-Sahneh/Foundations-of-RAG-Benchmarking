import os
from beir import util
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_scifact():
    """
    Downloads the SciFact dataset using the BEIR utility.
    """
    dataset_name = "scifact"
    out_dir = "datasets"
    
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Construct the URL for the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    
    try:
        logger.info(f"Starting download of '{dataset_name}' dataset...")
        data_path = util.download_and_unzip(url, out_dir)
        logger.info(f"Dataset downloaded and extracted to: {data_path}")
        logger.info("Data download script finished successfully.")
    except Exception as e:
        logger.error(f"An error occurred during download or extraction: {e}")

if __name__ == "__main__":
    download_scifact()