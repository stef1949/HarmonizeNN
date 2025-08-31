import pandas as pd
import numpy as np

# --- Configuration ---
N_SAMPLES = 200
N_GENES = 5000
N_DIFF_EXP_GENES = 100  # Genes with a true biological signal
N_BATCH_EFFECT_GENES = 1000 # Genes affected by the batch effect

print("Generating synthetic RNA-seq data...")

# --- 1. Create Metadata ---
samples = [f"Sample_{i+1}" for i in range(N_SAMPLES)]
# Assign samples to two batches (e.g., different sequencing centers)
batches = ['Batch1'] * int(N_SAMPLES * 0.6) + ['Batch2'] * int(N_SAMPLES * 0.4)
# Assign samples to two biological conditions
conditions = (['Tumor'] * int(N_SAMPLES * 0.3) + ['Normal'] * int(N_SAMPLES * 0.3) +
              ['Tumor'] * int(N_SAMPLES * 0.2) + ['Normal'] * int(N_SAMPLES * 0.2))

# Shuffle assignments to mix conditions across batches
np.random.shuffle(batches)
np.random.shuffle(conditions)

metadata = pd.DataFrame({
    'sample': samples,
    'batch': batches,
    'condition': conditions
})

print(f"Generated metadata with {len(metadata)} samples.")
print(pd.crosstab(metadata['batch'], metadata['condition']))

# --- 2. Create Gene Counts Data ---
gene_names = [f"Gene_{i+1}" for i in range(N_GENES)]

# Start with a baseline of random count data
counts = np.random.negative_binomial(n=20, p=0.5, size=(N_SAMPLES, N_GENES))

# a) Introduce the true biological signal (condition effect)
is_tumor = (metadata['condition'] == 'Tumor').values
diff_exp_indices = np.random.choice(N_GENES, N_DIFF_EXP_GENES, replace=False)

# Use np.ix_ to correctly select the rectangular block of (tumor_samples x diff_exp_genes)
tumor_row_indices = np.where(is_tumor)[0]
slicer_bio = np.ix_(tumor_row_indices, diff_exp_indices)
counts[slicer_bio] *= np.random.randint(2, 5, size=(is_tumor.sum(), N_DIFF_EXP_GENES))
print(f"Introduced biological signal in {N_DIFF_EXP_GENES} genes.")


# b) Introduce a strong batch effect
is_batch2 = (metadata['batch'] == 'Batch2').values
batch_effect_indices = np.random.choice(N_GENES, N_BATCH_EFFECT_GENES, replace=False)

# Apply the same fix for the batch effect
batch2_row_indices = np.where(is_batch2)[0]
slicer_batch = np.ix_(batch2_row_indices, batch_effect_indices)
counts[slicer_batch] += np.random.randint(50, 150, size=(is_batch2.sum(), N_BATCH_EFFECT_GENES))
print(f"Introduced a strong batch effect in {N_BATCH_EFFECT_GENES} genes.")

# Create the final DataFrame
counts_df = pd.DataFrame(counts, index=samples, columns=gene_names)
counts_df.index.name = 'sample_id'
counts_df.reset_index(inplace=True)


# --- 3. Save to CSV Files ---
counts_df.to_csv('synthetic_counts.csv', index=False)
metadata.to_csv('synthetic_metadata.csv', index=False)

print("\nSuccessfully created 'synthetic_counts.csv' and 'synthetic_metadata.csv'!")
print("You can now use these files to test the batch correction script.")