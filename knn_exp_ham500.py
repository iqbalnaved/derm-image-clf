# --------------------------------------------
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel

# ------------------ CONFIG ------------------
# ------------------ CONFIG ------------------
train_image_dir = "/mnt/d/Naved/Data/HAM500/split/train"       # <-- update
test_image_dir = "/mnt/d/Naved/Data/HAM500/split/test"       # <-- update
train_csv_path = "/mnt/d/Naved/Data/HAM500/demo.csv"    # <-- update
test_csv_path = "/mnt/d/Naved/Data/HAM500/test.csv"    # <-- update
model_name = 'google/vit-huge-patch14-224-in21k' #'google/vit-base-patch16-224-in21k'  # or any other HF vision model
plot_save_path = "/mnt/d/Naved/Outputs/ham500/plots"
dbname = 'HAM500'
ks = list(range(1,6)) # 1..5

# --------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()

# Read CSVs
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

filename_col = train_df.columns[0]
label_columns = train_df.columns[1:]

train_df['filename'] = train_df[filename_col]
train_df['label'] = train_df[label_columns].idxmax(axis=1)

test_df['filename'] = test_df[test_df.columns[0]]
test_df['label'] = test_df[label_columns].idxmax(axis=1)

def extract_features(df_subset, image_dir):
    features, labels = [], []

    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Extracting from {image_dir}"):
        label = row['label']
        image_path = os.path.join(image_dir, label, row['filename'])  # image path includes class subfolder

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {row['filename']} due to error: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        features.append(embedding)
        labels.append(label)

    return np.array(features), np.array(labels)

# Feature extraction
X_train, y_train = extract_features(train_df, train_image_dir)
X_test, y_test = extract_features(test_df, test_image_dir)

class_names = np.unique(np.concatenate([y_train, y_test]))
y_test_bin = label_binarize(y_test, classes=class_names)

def evaluate_knn(X_train, y_train, X_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro")

    y_pred_bin = label_binarize(y_pred, classes=class_names)
    auc = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovo")

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FN + FP)
    sensitivity = np.mean(TP / (TP + FN + 1e-8))
    specificity = np.mean(TN / (TN + FP + 1e-8))

    return acc, sensitivity, specificity, prec, f1, auc, cm, y_pred

# Run single evaluation for each k
acc_list = []

for k in ks:
    print(f"\n=== Evaluating k={k} ===")
    acc, sens, spec, prec, f1, auc, cm, y_pred = evaluate_knn(X_train, y_train, X_test, y_test, k)
    acc_list.append(acc)

    print(f"Accuracy     : {acc:.4f}")
    print(f"Sensitivity  : {sens:.4f}")
    print(f"Specificity  : {spec:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC (macro)  : {auc:.4f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix ({dbname}, k={k})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_path, f"{dbname}_conf_matrix_k{k}.png"))
    plt.show()

# Plot accuracy vs k
plt.figure(figsize=(6, 4))
plt.bar([str(k) for k in ks], acc_list)
plt.title(f"{dbname} | k vs Accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plot_save_path, f"{dbname}_knn_accuracy_plot.png"))
plt.show()