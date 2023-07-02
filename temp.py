import os
import shutil

def copy_txt_files(source_folder, target_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if not file.endswith((".txt", ".pdf", ".xlsx", ".docx")):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)
                shutil.copy(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

# Specify the source folder where the files are currently located
source_folder = r"C:\Users\DSi\Desktop\Projects\chatgpt-retrieval\data"

# Specify the target folder where you want to copy the .txt files
target_folder = r"C:\Users\DSi\Documents\data\others"

# Call the function to copy the .txt files
copy_txt_files(source_folder, target_folder)
