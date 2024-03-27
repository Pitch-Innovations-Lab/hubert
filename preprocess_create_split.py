# Import necessary libraries
import os
import shutil
from sklearn.model_selection import train_test_split

# Path to the file containing transcripts
transcripts_path = "/home/arjbah/Projects/pitch_innovations/hubert-finetune/wavs/transcripts.txt"

# Initialize a dictionary to hold the wav filenames and their corresponding transcripts
transcript_dict = {}

# Open the transcript file and read its contents
with open(transcripts_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Split each line into filename and transcript parts
        parts = line.strip().split(' ', 1)
        # Ensure the line is correctly formatted with two parts
        if len(parts) == 2:
            # Map the filename to its transcript in the dictionary
            transcript_dict[parts[0]] = parts[1]

# Split the dictionary keys into training and testing sets, 80% for training, 20% for testing
keys = list(transcript_dict.keys())
train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)

# Define a function to save wav files and their transcripts into specified folders
def save_data(folder_name, keys):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    # Define the path for the new transcript file within this folder
    transcript_file_path = os.path.join(folder_name, "transcripts.txt")
    # Open the new transcript file in write mode
    with open(transcript_file_path, 'w', encoding='utf-8') as transcript_file:
        for key in keys:
            # Copy each wav file into the new folder
            shutil.copy(f"/home/arjbah/Projects/pitch_innovations/hubert-finetune/audio-wav-16k-mono/{key}.wav", folder_name)
            # Write the corresponding transcript into the new transcript file
            transcript_file.write(f"{key} {transcript_dict[key]}\n")

# Save the training data (80% of files) into a folder named "train"
save_data("train", train_keys)
# Save the testing data (20% of files) into a folder named "dev"
save_data("dev", test_keys)