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
    """Convert scientific names to valid directory names"""
    if pd.isna(name):
        return "unknown"
    # Convert to lowercase and replace spaces with underscores
    name = str(name).lower().strip()
    name = re.sub(r"\s+", "_", name)
    # Remove any characters that aren't suitable for directory names
    name = re.sub(r"[^\w_-]", "", name)
    return name


def create_directory_structure(df, base_dir="dataset"):
    """Create the directory structure based on species"""
    # Create main dataset directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logging.info(f"Created base directory: {base_dir}")

    # Get unique scientific names and create directories
    unique_species = df["scientific_name"].dropna().unique()

    created_dirs = 0
    for species in unique_species:
        clean_name = clean_directory_name(species)
        species_dir = os.path.join(base_dir, clean_name)
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)
            created_dirs += 1

    logging.info(f"Created {created_dirs} species directories")
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
        return False


def download_all_images(df, base_dir="dataset", delay=0.15):
    """Download all images from the dataframe"""
    success_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        if pd.isna(row["scientific_name"]) or pd.isna(row["image_url"]):
            continue

        clean_name = clean_directory_name(row["scientific_name"])
        species_dir = clean_name

        # Create a unique filename using the scientific name and observation ID
        try:
            obs_id = row["url"].split("/")[-1]
            filename = f"{clean_name}_{obs_id}.jpg"
        except:
            # Fallback to index if URL parsing fails
            filename = f"{clean_name}_image_{idx}.jpg"

        save_path = os.path.join(base_dir, species_dir, filename)

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
    """Process the data and download images"""
    logging.info(f"Reading data from {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)

    # Keep only necessary columns for memory efficiency if specified
    if keep_cols:
        df = df[keep_cols]

    # Create directory structure
    base_dir = create_directory_structure(df, output_dir)

    # Download images
    logging.info("Starting image downloads")
    success_count, error_count = download_all_images(df, base_dir, delay)

    logging.info(
        f"Download complete. Successfully downloaded {success_count} images. Failed: {error_count}"
    )


def main():
    """Main function to handle command line arguments and run the program"""
    parser = argparse.ArgumentParser(
        description="Download nature observation images by species"
    )
    parser.add_argument("--csv_file", help="Path to the observations CSV file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dataset",
        help="Directory to save images (default: dataset)",
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
