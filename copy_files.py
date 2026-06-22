import os
import shutil
import pandas as pd
import os
import shutil
import pandas as pd

def copy_images_by_labels(csv_path, source_dir, dest_dirs):
    """
    csv_path: Path to the CSV file
    source_dir: Directory containing source images
    dest_dirs: Dict mapping class name (from CSV columns) to destination directory
               e.g., {'melanoma': 'path/to/melanoma_dir', 'benign': 'path/to/benign_dir'}
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure 'filename' column exists
    if 'filename' not in df.columns:
        raise ValueError("CSV must contain a 'filename' column.")

    # Get class names from the remaining columns
    class_names = [col for col in df.columns if col != 'filename']

    for _, row in df.iterrows():
        filename = row['filename']
        src_path = os.path.join(source_dir, filename)

        if not os.path.isfile(src_path):
            print(f"File not found: {filename}")
            continue

        # Check each class column
        for class_name in class_names:
            if row[class_name] == 1:
                dest_dir = dest_dirs.get(class_name)
                if dest_dir:
                    os.makedirs(dest_dir, exist_ok=True)
                    dst_path = os.path.join(dest_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied {filename} to {class_name} directory")
                else:
                    print(f"No destination directory provided for class: {class_name}")

if __name__ == "__main__":
    # Example usage
    csv_path = "your_file.csv"
    source_dir = "path/to/source_images"
    dest_dirs = {
        'melanoma': 'path/to/melanoma_dest',
        'benign': 'path/to/benign_dest'
    }

    copy_images_by_labels(csv_path, source_dir, dest_dirs)



if __name__ == "__main__":
    # Example usage
    csv_path = "/mnt/d/Naved/Codes/ManyICL/ManyICL/dataset/HAM10K_EXP2_rep1/test.csv"
    source_dir = "/mnt/d/Naved/Codes/ManyICL/ManyICL/dataset/HAM10K_EXP2_rep1/images"
    dest_dir_class1 = "path/to/destination_dir/mm_images"
    dest_dir_class2 = "path/to/destination_dir/nv_images"

    copy_images_from_csv(csv_path, source_dir, dest_dir)
