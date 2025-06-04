import os
import json
import logging
from typing import List, Dict, Tuple
from datetime import datetime


def has_raw_files(directory: str) -> bool:
    """
    Check if a directory contains any raw files.

    Args:
        directory: Directory path to check

    Returns:
        True if directory contains raw files, False otherwise
    """
    raw_extensions = {".nef"}
    try:
        for item in os.listdir(directory):
            if os.path.splitext(item.lower())[1] in raw_extensions:
                return True
    except (PermissionError, Exception):
        pass
    return False


def get_all_subdirectories(
    root_dir: str, processed_dirs: Dict[str, str], logger: logging.Logger = None
) -> List[Tuple[str, bool]]:
    """
    Recursively get all subdirectories that contain raw files.
    Returns a list of tuples containing directory paths and their processing status.

    Args:
        root_dir: The root directory to start crawling from
        processed_dirs: Dictionary of processed directories and their timestamps
        logger: Optional logger instance for logging directory access issues

    Returns:
        List of tuples (directory_path, is_processed)
    """
    result = []
    try:
        # Check if current directory has raw files
        if has_raw_files(root_dir):
            is_processed = root_dir in processed_dirs
            result.append((root_dir, is_processed))

        # Check all subdirectories
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                subdirs = get_all_subdirectories(item_path, processed_dirs, logger)
                result.extend(subdirs)

    except PermissionError:
        if logger:
            logger.warning(f"Warning: No permission to access {root_dir}, skipping...")
    except Exception as e:
        if logger:
            logger.warning(f"Warning: Error accessing {root_dir}: {str(e)}, skipping...")
    return result


def load_processed_directories(processed_file: str, logger: logging.Logger = None) -> Dict[str, str]:
    """
    Load the list of previously processed directories and their timestamps.

    Args:
        processed_file: Path to the JSON file containing processed directories
        logger: Optional logger instance for logging errors

    Returns:
        Dictionary mapping directory paths to their processing timestamps
    """
    if os.path.exists(processed_file):
        try:
            with open(processed_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if logger:
                logger.warning(f"Error reading {processed_file}, starting with empty processed list")
            return {}
    return {}


def save_processed_directory(directory: str, processed_file: str, logger: logging.Logger = None) -> None:
    """
    Save a directory to the list of processed directories with current timestamp.

    Args:
        directory: Directory path to mark as processed
        processed_file: Path to the JSON file to save processed directories
        logger: Optional logger instance for logging errors
    """
    try:
        processed = load_processed_directories(processed_file, logger)
        processed[directory] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(processed_file, "w") as f:
            json.dump(processed, f, indent=2)
    except Exception as e:
        if logger:
            logger.error(f"Error saving processed directory {directory}: {str(e)}")
