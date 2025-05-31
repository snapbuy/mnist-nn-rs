#!/usr/bin/env python3
import os
import json
import subprocess
import sys
import gzip
import numpy as np
import pandas as pd
import struct
import csv
from pathlib import Path

# Create the mnistdata directory if it doesn't exist
os.makedirs('mnistdata', exist_ok=True)
os.makedirs('mnistdata/raw', exist_ok=True)

def check_kaggle_credentials():
    """Check if Kaggle API credentials are available"""
    kaggle_dir = os.path.join(str(Path.home()), '.kaggle')
    kaggle_cred = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_cred):
        print("Kaggle API credentials not found.")
        print("\nTo use the Kaggle API, you need to:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to your account settings (https://www.kaggle.com/account)")
        print("3. Scroll down to the 'API' section and click 'Create New API Token'")
        print("4. This will download a kaggle.json file")
        print(f"5. Create the directory: {kaggle_dir}")
        print(f"6. Move the downloaded kaggle.json file to {kaggle_cred}")
        print(f"7. Ensure file permissions are secure: chmod 600 {kaggle_cred}")
        print("\nAfter setting up your Kaggle credentials, run this script again.")
        return False
    
    return True

def download_mnist_from_kaggle():
    """Download MNIST dataset from Kaggle"""
    print("Downloading MNIST dataset from Kaggle...")
    try:
        # Download the dataset
        subprocess.run(["kaggle", "datasets", "download", "oddrationale/mnist-in-csv", "-p", "mnistdata/raw"], 
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Unzip the downloaded file
        subprocess.run(["unzip", "-o", "mnistdata/raw/mnist-in-csv.zip", "-d", "mnistdata/raw"], 
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("MNIST dataset downloaded and extracted successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading MNIST dataset: {e}")
        print(f"Command output: {e.stdout.decode() if e.stdout else ''}")
        print(f"Error output: {e.stderr.decode() if e.stderr else ''}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def process_kaggle_csv_files():
    """Process the CSV files downloaded from Kaggle and copy them to the required location"""
    try:
        # Check if the files exist
        train_file = 'mnistdata/raw/mnist_train.csv'
        test_file = 'mnistdata/raw/mnist_test.csv'
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print("Expected CSV files not found in the extracted data.")
            return False
        
        # Copy the files to the required location
        os.replace(train_file, 'mnistdata/mnist_train.csv')
        os.replace(test_file, 'mnistdata/mnist_test.csv')
        
        print("CSV files processed and moved to the correct location.")
        return True
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        return False

def download_mnist_manually():
    """Provide instructions for manual download as a fallback"""
    print("\nAutomatic download failed. Please follow these instructions to manually download the MNIST dataset:")
    print("1. Go to https://www.kaggle.com/oddrationale/mnist-in-csv")
    print("2. Download the dataset (you'll need a Kaggle account)")
    print("3. Extract the zip file")
    print("4. Copy mnist_train.csv and mnist_test.csv to the 'mnistdata' directory in this project")
    print("\nAfter downloading the files manually, run 'cargo run' to train the neural network.")
    return False

def main():
    print("MNIST Dataset Downloader")
    print("=======================")
    
    # First, remove any potentially corrupted files
    for file_type in ['train', 'test']:
        file_path = f'mnistdata/mnist_{file_type}.csv'
        if os.path.exists(file_path):
            print(f"Removing existing file: {file_path}")
            os.remove(file_path)
    
    # Check for Kaggle credentials
    if not check_kaggle_credentials():
        return download_mnist_manually()
    
    # Download MNIST dataset from Kaggle
    if not download_mnist_from_kaggle():
        return download_mnist_manually()
    
    # Process the CSV files
    if not process_kaggle_csv_files():
        return download_mnist_manually()
    
    print("\nMNIST dataset downloaded and prepared successfully.")
    print("Training set saved to mnistdata/mnist_train.csv")
    print("Test set saved to mnistdata/mnist_test.csv")
    print("\nNow you can run 'cargo run' to train the neural network.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

