"""File operations for DONUT project.
This module contains functions for handling file operations 
such as listing files in a directory."""

from pathlib import Path
import csv

def list_files_in_directory(directory):
    """
    List all files in the given directory.

    Args:
        directory (str): Path to the directory.
    Returns:
        List of file paths in the directory.
    """
    p = Path(directory)
    return [str(file) for file in p.iterdir() if file.is_file()]

def write_to_csv(*cols, filename="output.csv", header=None):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # optional header
        if header is not None:
            writer.writerow(header)
        # transpose columns → rows
        for row in zip(*cols):
            writer.writerow(row)