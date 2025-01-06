import os

# Set the directory path
directory_path = "/home/data/uieb"

# Set the path for the output text file
output_file_path = "/home/data/uieb/data_list.txt"

# List all files in the directory
all_files = os.listdir(directory_path)

# Filter out directories, keep only files
file_names = [f for f in all_files if os.path.isfile(os.path.join(directory_path, f))]

# Write the file names to a text file
with open(output_file_path, 'w') as file:
    for name in file_names:
        file.write(name + '\n')

print(f"File names saved to {output_file_path}")