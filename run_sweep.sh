#!/bin/bash

# --- Hyperparameter Sweep Configuration ---

# 1. Learning Rates to test
learning_rates="0.001 0.0005"

# 2. Standard Architectures to test (for use_residual=False)
standard_architectures=(
    "256,128,64"
    "512,256"
)

# 3. Residual Architectures to test (for use_residual=True)
#    Note: All hidden layers must be the same size for residual connections.
residual_architectures=(
    "256,256,256"
    "512,512"
)

# --- End of Configuration ---


echo "Starting hyperparameter sweep..."

# --- Sweep for Standard Networks ---
echo "--- Testing Standard Architectures (use_residual=False) ---"
for lr in $learning_rates; do
    for arch in "${standard_architectures[@]}"; do
        # Create a descriptive model name
        model_name="model_std_lr${lr}_arch${arch//,/-}.pt"
        echo "TRAINING: lr=$lr, arch=[$arch], residual=False. Saving to $model_name"

        # Run the training command
        python NN_batch_correct.py \
            --counts train_data.csv \
            --metadata train_meta.csv \
            --lr "$lr" \
            --enc_hidden "$arch" \
            --epochs 50 \
            --save_model "$model_name"
    done
done


# --- Sweep for Residual Networks ---
echo "--- Testing Residual Architectures (use_residual=True) ---"
for lr in $learning_rates; do
    for arch in "${residual_architectures[@]}"; do
        # Create a descriptive model name
        model_name="model_res_lr${lr}_arch${arch//,/-}.pt"
        echo "TRAINING: lr=$lr, arch=[$arch], residual=True. Saving to $model_name"

        # Run the training command with the --use_residual flag
        python NN_batch_correct.py \
            --counts train_data.csv \
            --metadata train_meta.csv \
            --lr "$lr" \
            --enc_hidden "$arch" \
            --use_residual \
            --epochs 50 \
            --save_model "$model_name"
    done
done

echo "Hyperparameter sweep complete."