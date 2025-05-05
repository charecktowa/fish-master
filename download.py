#!/usr/bin/env python3
import os
import requests
import pandas as pd
import time
import re
import argparse
from tqdm import tqdm
import logging


def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def clean_directory_name(name):
    """Convert scientific names to valid parts of filenames"""
    if pd.isna(name):
        return "unknown"
    # Convert to lowercase and replace spaces with underscores
    name = str(name).lower().strip()
    name = re.sub(r"\s+", "_", name)
    # Remove any characters that aren't suitable for filenames (adjust as needed)
    name = re.sub(r"[^\w_-]", "", name)
    return name


def ensure_base_directory(base_dir="dataset"):
    """Ensure the base directory exists"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logging.info(f"Created base directory: {base_dir}")
    else:
        logging.info(f"Base directory already exists: {base_dir}")
    return base_dir


def download_image(url, save_path):
    """Download an image from URL to save_path"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        # Attempt to remove partially downloaded file if error occurs
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError as oe:
                logging.error(f"Error removing partial file {save_path}: {oe}")
        return False


def download_all_images(df, base_dir="dataset", delay=0.15):
    """Download all images from the dataframe into the base directory"""
    success_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        # Skip rows with missing scientific name or image URL
        if pd.isna(row["scientific_name"]) or pd.isna(row["image_url"]):
            logging.debug(f"Skipping row {idx} due to missing data.")
            continue

        clean_name = clean_directory_name(row["scientific_name"])
        # Use the dataframe index as the file number for uniqueness
        file_number = idx

        # Create a unique filename using the scientific name and file number
        try:
            # Optionally, try to include observation ID for more context
            obs_id_part = ""
            try:
                # Extract a unique identifier from the URL, assuming it's the last part
                obs_id = str(row["url"]).split("/")[-1]
                # Ensure obs_id is valid for filename, remove non-alphanumeric
                obs_id = re.sub(r"\W+", "", obs_id)
                if obs_id:  # If a valid obs_id was extracted
                    obs_id_part = f"_{obs_id}"  # Include it in the filename
            except Exception as e:
                # Log if obs_id extraction fails but continue with index
                logging.debug(
                    f"Could not parse observation ID from URL {row['url']} for row {idx}: {e}. Using index only."
                )

            # Construct filename including the file number (index)
            filename = f"{clean_name}{obs_id_part}_{file_number}.jpg"

        except Exception as e:
            # Fallback filename using only index if any other error occurs
            logging.warning(
                f"Error generating filename for row {idx}: {e}. Using basic index format."
            )
            filename = f"{clean_name}_image_{file_number}.jpg"

        # Define the full path to save the image in the base directory
        save_path = os.path.join(base_dir, filename)

        # Skip if file already exists
        if os.path.exists(save_path):
            logging.debug(f"Skipping existing file: {save_path}")
            continue

        # Download the image
        if download_image(row["image_url"], save_path):
            success_count += 1
        else:
            error_count += 1

        # Add a small delay to avoid overloading the server
        time.sleep(delay)

    return success_count, error_count


def process_data(csv_file, output_dir="dataset", delay=0.15, keep_cols=None):
    """Process the data and download images into a single directory"""
    logging.info(f"Reading data from {csv_file}")
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_file}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}: {e}")
        return

    # Check if necessary columns exist
    if keep_cols:
        missing_cols = [col for col in keep_cols if col not in df.columns]
        if missing_cols:
            logging.error(
                f"Error: Missing required columns in CSV: {', '.join(missing_cols)}"
            )
            return
        # Keep only necessary columns for memory efficiency if specified
        df = df[keep_cols]

    # Ensure the base output directory exists
    base_dir = ensure_base_directory(output_dir)

    # Download images
    logging.info(f"Starting image downloads into {base_dir}")
    success_count, error_count = download_all_images(df, base_dir, delay)

    logging.info(
        f"Download complete. Successfully downloaded {success_count} images. Failed: {error_count}"
    )


def main():
    """Main function to handle command line arguments and run the program"""
    parser = argparse.ArgumentParser(
        description="Download nature observation images by species into a single directory"
    )
    parser.add_argument(
        "--csv_file", help="Path to the observations CSV file"
    )  # Made csv_file mandatory
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dataset",
        help="Directory to save all images (default: dataset)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.15,
        help="Delay between downloads in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Required columns for the download process
    necessary_columns = ["scientific_name", "image_url", "url"]

    # Process data and download images
    process_data(
        args.csv_file,
        output_dir=args.output_dir,
        delay=args.delay,
        keep_cols=necessary_columns,
    )


if __name__ == "__main__":
    main()
