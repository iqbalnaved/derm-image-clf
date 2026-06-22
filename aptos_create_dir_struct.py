import os
import shutil
import pandas as pd

def split_images_by_ratio(csv_path, src_dir, dest_dir, train_ratio=0.8, random_state=42):
    """
    Splits each class into train/test using given ratio.
    Saves images under dest_dir/train/<class>/ and dest_dir/test/<class>/.
    All copied files are renamed with .png extension.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: filename and label")

    df = df.iloc[:, :2]  # first two columns only
    df.columns = ["filename", "label"]

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    for label, group in df.groupby("label"):
        n_total = len(group)
        n_train = int(n_total * train_ratio)
        
        train_samples = group.iloc[:n_train]
        test_samples = group.iloc[n_train:]

        # Copy train
        for fname in train_samples["filename"]:
            src_path = os.path.join(src_dir, fname + '.png')
            if not os.path.isfile(src_path):
                print(f"⚠️ Missing file: {src_path}")
                continue

            
            label_dir = os.path.join(dest_dir, "train", str(label))
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy2(src_path, os.path.join(label_dir, fname + '.png'))

        # Copy test
        for fname in test_samples["filename"]:
            src_path = os.path.join(src_dir, fname + '.png')
            if not os.path.isfile(src_path):
                print(f"⚠️ Missing file: {src_path}")
                continue

            label_dir = os.path.join(dest_dir, "test", str(label))
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy2(src_path, os.path.join(label_dir, fname + '.png'))

    print("✅ 80/20 train-test split completed.")


# ==== Example usage ====
# if __name__ == "__main__":
    # csv_path = "/path/to/labels.csv"
    # src_dir = "/path/to/source_images"
    # dest_dir = "/path/to/new_dataset"

    # split_images_by_ratio(csv_path, src_dir, dest_dir, train_ratio=0.8)




# ==== Example usage ====
if __name__ == "__main__":

    csv_path = "/mnt/d/Naved/Data/APTOS2019/aptos2019-blindness-detection/train.csv"
    src_dir = "/mnt/d/Naved/Data/APTOS2019/aptos2019-blindness-detection/train_images"
    dest_dir = "/mnt/d/Naved/Data/APTOS2019/atpos2019_sampled"

    split_images_by_ratio(csv_path, src_dir, dest_dir, train_ratio=0.8)

