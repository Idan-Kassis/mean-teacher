import os

# Function to get the list of files in a folder
def get_files_in_folder(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            files.append(file)
    return files

# Function to write data to the text file
def write_to_txt(filename, folder, f):
    f.write(f"{filename} {folder}\n")

# Folder path where your files are located
data_path = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Train"

# Output file path
classes = ["Negative", "Positive"]
output_file = "/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/labels.txt"

with open(output_file, "w") as f:
    for category in classes:
        folder_path = os.path.join(data_path, category)
        for file in get_files_in_folder(folder_path):
            # Get the folder name from the folder path
            folder_name = os.path.basename(folder_path)
            write_to_txt(file, folder_name, f)

print("Data written to output.txt successfully.")
