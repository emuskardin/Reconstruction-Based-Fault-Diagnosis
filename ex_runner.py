import os
import subprocess
import sys

# Path to the training_data folder
training_data_folder = 'data/training_data'

# Loop through all files in the folder
for filename in os.listdir(training_data_folder):
    file_path = os.path.join(training_data_folder, filename)

    # Check if it is a file (and not a subdirectory)
    if os.path.isfile(file_path):
        print(f"Running diagnoser on: {file_path}")

        # Run the command
        subprocess.run([sys.executable, 'RunDiagnoser.py', file_path])