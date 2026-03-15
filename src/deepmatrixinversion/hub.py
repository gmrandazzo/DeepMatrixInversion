"""hub.py

This module handles downloading and managing models from the Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model_from_hub(repo_id: str):
    """
    Downloads a model ensemble from the Hugging Face Hub.
    
    Args:
        repo_id (str): The repository ID on Hugging Face (e.g., 'user/dmx-3x3-resnet')
        
    Returns:
        str: The local path to the downloaded model directory.
    """
    print(f"Checking for model '{repo_id}' on Hugging Face Hub...")
    try:
        # Download the snapshot of the entire model directory (ensemble)
        local_dir = snapshot_download(repo_id=repo_id)
        print(f"Model downloaded/verified at: {local_dir}")
        return local_dir
    except Exception as e:
        raise ValueError(f"Could not download model from Hugging Face Hub: {e}")
