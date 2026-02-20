"""File operations for DONUT project.
This module contains functions for handling file operations 
such as listing files in a directory."""

from pathlib import Path

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
