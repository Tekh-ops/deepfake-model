# metadata.py

# Import required modules
from datetime import datetime
from modules.metadata_analysis.hashing import generate_hash   # reuse hashing function


def create_metadata(file_path, result, confidence):
    """
    This function creates structured metadata for a given file.

    Parameters:
    - file_path: path to the input file
    - result: prediction result ("Real" or "Fake")
    - confidence: confidence score (float)

    Returns:
    - metadata dictionary
    """

    # Extract only file name from full path
    file_name = file_path.split("/")[-1]

    # Generate SHA-256 hash of the file
    file_hash = generate_hash(file_path)

    # Get current timestamp in readable format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create metadata dictionary (matches contract.json structure)
    metadata = {
        "file_name": file_name,
        "result": result,
        "confidence": confidence,
        "hash": file_hash,
        "timestamp": timestamp
    }

    # Return metadata
    return metadata