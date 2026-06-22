# Linear Evaluation on Image Classification Tasks
# Training and evaluation using the PAD-UFES dataset as an example. Replace the CSV path and root path with your own dataset.

# Key Parameters
# batch_size: Adjust based on the memory size of your GPU.
# model: Model size - "PanDerm_Large_LP" (original paper model) or "PanDerm_Base_LP" (smaller version)
# nb_classes: Set this to the number of classes in your evaluation dataset.
# percent_data: Controls the percentage of training data used. For example, 0.1 means evaluate models using 10% of the training data. Modify this if you want to conduct label efficiency generalization experiments.
# csv_path: Organize your dataset as described in the "Data Preparation" section.
# root_path: The path of your folder for saved images.
# pretrained_checkpoint: Path to the pretrain checkpoint - "panderm_ll_data6_checkpoint-499.pth" for "PanDerm_Large_LP" and "panderm_bb_data6_checkpoint-499.pth" for "PanDerm_Base_LP".


cd classification
CUDA_VISIBLE_DEVICES=1 python3 linear_eval.py \
  --batch_size 1000 \
  --model "PanDerm_Large_LP" \
  --nb_classes 6 \
  --percent_data 1.0 \
  --csv_filename "PanDerm_Large_LP_result.csv" \
  --output_dir "/path/to/your/PanDerm/output_dir/PanDerm_res/" \
  --csv_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/2000.csv" \
  --root_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/images/" \
  --pretrained_checkpoint "/path/to/your/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"


# To run the evaluations:

cd classification
bash script/lp_reproduce.sh


#-----------------------------------

# Fine-tuning on Image Classification Tasks
# Key Parameters
# model: Model size - "PanDerm_Large_FT" (original paper model) or "PanDerm_Base_FT" (smaller version)
# pretrained_checkpoint: Path to the pretrain checkpoint - "panderm_ll_data6_checkpoint-499.pth" for "PanDerm_Large_FT" and "panderm_bb_data6_checkpoint-499.pth" for "PanDerm_Base_FT".
# nb_classes: Set this to the number of classes in your evaluation dataset.
# weights: Setting to use the weighted random sampler for the imbalanced class dataset.
# monitor: Choosing your checkpoint based on "acc" or "recall".
# csv_path: Organize your dataset as described in the "Data Preparation" section.
# root_path: The path of your folder for saved images.
# TTA: Enable Test-Time Augmentation. You can modify the augmentation setting in the class TTAHandler classification/furnace/engine_for_finetuning.py. -- eval: Model inference.
# Recommended Configuration for fine-tuning
# Our experiments show the following hyperparameters deliver optimal performance across various evaluation datasets:

# Batch size: 128
# Learning rate: 5e-4
# Training epochs: 50
# Enable the weighted random sampler
# Enable TTA during testing
# We observed that the hyperparameter setting is robust across datasets and typically doesn't require adjustment.

# Start Training
# You could fine-tune PanDerm on your dataset. Here is a command-line example for fine-tuning PanDerm_Large on the PAD-UFES dataset:

MODEL_NAME="PanDerm_Large_FT"
MODEL_PATH="/path/to/your/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"

CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model $MODEL_NAME \
    --pretrained_checkpoint $MODEL_PATH \
    --nb_classes 6 \
    --batch_size 128 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --weights \
    --sin_pos_emb \
    --no_auto_resume \
    --exp_name exp_name "pad finetune and eval" \
    --imagenet_default_mean_and_std \
    --wandb_name "Reproduce_PAD_FT_${seed}" \
    --output_dir /path/to/your/PanDerm/Evaluation_datasets/PAD_Res/ \ # Your best epoch's fine-tuned checkpoint and model output results on the test set will be saved in this directory
    --csv_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/2000.csv" \
    --root_path "/path/to/your/PanDerm/Evaluation_datasets/pad-ufes/images/ " \
    --seed 0 
    
# script for finetuning :     
cd classification
bash script/finetune_train.sh     

# script for evaluation : 
cd classification
bash script/finetune_test.sh