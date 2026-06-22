import os
import shutil
import pandas as pd

def copy_files_to_label_dirs(csv_file, source_dir, dest_dir):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Extract file names and one-hot labels
    filenames = df.iloc[:, 0]
    one_hot_labels = df.iloc[:, 1:]
    class_names = one_hot_labels.columns.tolist()

    for idx, filename in enumerate(filenames):
        label_row = one_hot_labels.iloc[idx]
        if 1 not in label_row.values:
            print(f"Warning: No label found for file {filename}. Skipping.")
            continue

        # Get class name where value is 1
        class_index = label_row.values.argmax()
        class_name = class_names[class_index]

        src_path = os.path.join(source_dir, filename)
        class_dir = os.path.join(dest_dir, class_name)
        dest_path = os.path.join(class_dir, filename)

        os.makedirs(class_dir, exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename} → {class_name}/")
        else:
            print(f"File not found: {filename}")

# Example usage
# csv_file = 'your_csv_file.csv'
# source_dir = 'path/to/source_directory'
# dest_dir = 'path/to/destination_directory'

# copy_files_to_label_dirs(csv_file, source_dir, dest_dir)


# csv_file = "/mnt/d/Naved/Data/HAM500/demo.csv"
# source_dir = "/mnt/d/Naved/Data/HAM500/images"
# dest_dir = "/mnt/d/Naved/Data/HAM500/train/"

csv_file = "/mnt/d/Naved/Data/HAM500/test.csv"
source_dir = "/mnt/d/Naved/Data/HAM500/images"
dest_dir = "/mnt/d/Naved/Data/HAM500/test/"

copy_files_to_label_dirs(csv_file, source_dir, dest_dir)
