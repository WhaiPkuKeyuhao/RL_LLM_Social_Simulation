import os
import re
import pickle

def seconds_to_dhms(total_seconds):
    day, remainder = divmod(total_seconds, 3600*24)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(day)+1, int(hours), int(minutes), int(seconds)

def create_folder_if_not_exists(path):
    """
    If the path does not exist, create a folder at that location.
    
    Args:
        path (str): The path to check and possibly create a folder at.
    
    Returns:
        str: The created folder's absolute path if it was created, otherwise the original path.
    """
    # Check if the path exists
    if not os.path.exists(path):
        # If not, try to create a new folder at that location
        try:
            os.makedirs(path)
            print(f"Created folder '{path}'")
            return path
        except OSError as e:
            print(f"Failed to create folder '{path}': {e}")
    else:
        print(f"'{path}' already exists. No action taken.")
    
    # If we reached this point, the path was either valid (and thus an existing file or folder) 
    # or creation of a new folder at that location failed.
    return path

def is_number(s):
    """
    Judge whether a string is a number.
    
    Args:
        s (str): The input string to be judged.
    
    Returns:
        bool: True if the string is a number, False otherwise.
    """
    pattern = r"^-?\d+(\.\d+)?$"
    return bool(re.match(pattern, s))
