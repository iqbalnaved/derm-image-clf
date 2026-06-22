# A dermoscopy image classification study on multimodal LLMs (GPT-4.x / GPT-5.x)

Dermoscopy image classification experiments using multimodal LLMs (GPT-4.x / GPT-5.x), vision transformer embeddings, and the PanDerm foundation model. Targets binary and multi-class lesion classification on HAM10000 and related datasets.

---

## Datasets

| Dataset | Task | Classes |
|---|---|---|
| HAM10000 (HAM500 split) | Binary dermoscopy classification | Melanoma vs. Nevus |
| APTOS 2019 | Diabetic retinopathy grading | Multi-class |
| PAD-UFES | Multi-class skin lesion classification | 6 classes |

---

## Repository Structure

```
derm-image-clf/
├── ham500.py                    # GPT-4.x / GPT-5.x zero-shot & few-shot inference on HAM500
├── knn_exp_ham500.py            # ViT embedding extraction + k-NN evaluation on HAM500
├── panderm_runs.py              # PanDerm linear probe & fine-tuning run commands
├── panderm_ham10k_sampled.py    # PanDerm evaluation on HAM10K sampled split
├── aptos_create_dir_struct.py   # Directory structure builder for APTOS dataset
├── copy_files.py                # Utility: copy image files by class
├── copy_files_reading_csv.py    # Utility: copy image files based on CSV labels
└── ham500.txt                   # HAM500 file list / metadata
```

---

## Experiments

### 1. LLM Few-Shot / Zero-Shot Classification (`ham500.py`)

Binary Melanoma vs. Nevus classification via OpenAI vision models.

**Models supported:** `gpt-4.1-2025-04-14`, `gpt-5`, `gpt-5-mini`

**Modes:**
- `0-shot`: image + chain-of-thought prompt only
- `k-shot` (k=1,3,5,7): balanced few-shot examples sampled from demo split; structured JSON output with `thoughts` + `answer` fields

**Metrics:** Accuracy, Sensitivity, Specificity

**Usage:**
```bash
export OPENAI_API_KEY=your_key

python ham500.py -s 3 -r 1 -m gpt-4.1-2025-04-14
# -s: shot count (0, 1, 3, 5, 7)
# -r: run replication number
# -m: model name
```

**Output:** `{dataset}_run{r}_{k}shot_{model}.txt` saved to configured `output_dir`

**Expected data layout:**
```
HAM10K_EXP2_rep1/
├── images/          # flat directory of .jpg dermoscopy images
├── demo.csv         # one-hot encoded; columns: image_id, mel, nv, ...
└── test.csv         # same format
```

---

### 2. ViT + k-NN Evaluation (`knn_exp_ham500.py`)

Extracts CLS-token embeddings from a pretrained ViT (`google/vit-huge-patch14-224-in21k`) via HuggingFace Transformers, then runs k-NN classification for k=1..5.

**Metrics:** Accuracy, Sensitivity, Specificity, Precision, F1, AUC (macro OvO)

**Outputs:** Per-k confusion matrix PNGs + accuracy-vs-k bar plot

**Usage:** Update the config block at the top of the file, then:
```bash
python knn_exp_ham500.py
```

**Expected data layout:**
```
HAM500/
├── split/
│   ├── train/{class_name}/image.jpg
│   └── test/{class_name}/image.jpg
├── demo.csv
└── test.csv
```

---

### 3. PanDerm Foundation Model (`panderm_runs.py`, `panderm_ham10k_sampled.py`)

Linear probe and fine-tuning commands for [PanDerm](https://github.com/SiyuanYan1/PanDerm) on dermoscopy datasets.

**Models:** `PanDerm_Large_LP` / `PanDerm_Base_LP` (linear probe), `PanDerm_Large_FT` / `PanDerm_Base_FT` (fine-tune)

**Linear probe:**
```bash
cd classification
CUDA_VISIBLE_DEVICES=0 python3 linear_eval.py \
  --model PanDerm_Large_LP \
  --nb_classes 6 \
  --percent_data 1.0 \
  --csv_path /path/to/dataset.csv \
  --root_path /path/to/images/ \
  --pretrained_checkpoint /path/to/panderm_ll_data6_checkpoint-499.pth
```

**Fine-tuning:**
```bash
cd classification
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
  --model PanDerm_Large_FT \
  --nb_classes 6 \
  --batch_size 128 --lr 5e-4 --epochs 50 \
  --weights --TTA \
  --csv_path /path/to/dataset.csv \
  --root_path /path/to/images/
```

Recommended hyperparameters (robust across datasets): batch=128, lr=5e-4, epochs=50, weighted random sampler, TTA enabled.

Pretrained weights: download from [PanDerm releases](https://github.com/SiyuanYan1/PanDerm).

---

## Installation

```bash
pip install openai pillow numpy pandas scikit-learn torch torchvision transformers tqdm matplotlib
```

GPU required for ViT feature extraction at scale. CPU fallback is available but slow.

---

## Environment Variables

```bash
export OPENAI_API_KEY=sk-...   # required for ham500.py
```

---

## Notes

- HAM500 is a balanced 500-image subset of HAM10000 (Melanoma/Nevus binary task).
- All LLM experiments use chain-of-thought prompting with forced JSON output format: `{"thoughts": "...", "answer": "Melanoma" | "Nevus"}`.
- Few-shot examples are randomly sampled per query image; use `-r` to replicate runs.
- Hardcoded paths in scripts (`/mnt/d/Naved/...`) must be updated before running.
