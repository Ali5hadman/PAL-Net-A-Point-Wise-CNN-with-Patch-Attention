import sys
from pathlib import Path

# Add the parent directory of the current working directory to sys.path
sys.path.insert(0, str(Path.cwd().parent))


import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import numpy as np
from tqdm import tqdm
from importlib import reload
import random
import matplotlib.pyplot as plt


print(torch.version.cuda)
print(torch.cuda.get_device_name(0))


# Set all seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)  # Sets seed for PyTorch CPU & CUDA operations
    torch.cuda.manual_seed_all(seed)  # Sets seed for all CUDA devices (if used)
    np.random.seed(seed)  # Sets seed for NumPy
    random.seed(seed)  # Sets seed for Python's built-in random module
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization that may introduce randomness
set_seed(12345)  # Use the same seed as in the model


#------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, random_split, DataLoader

from src.datasets.dataset import LafasDataset, mesh_collate_fn, ThresholdSampler
from src.datasets.patch_dataset import PatchDataset, PatchDataset_sampled, PatchDataset_distance_based
from src.utils.utils import get_patches, fix_prediction, fix_precition_average, compute_distance_error, compute_pointwise_error



df_results = pd.DataFrame()
point_wise_closes = []
point_wise_average = []
distance_wise_closes = []
distance_wise_average = []



# ——— Usage ———
dataset = LafasDataset(
    root_dirs=["dataset", "validation_set"],
    # root_dirs=["sample"],
    cache_dir="cache(processed)_npz"
)


# # 2.1) Filtering dataset (outliers of the registration process)
sampler = ThresholdSampler(dataset, threshold=120.0, ref_idx=0)
valid_indices = list(sampler)
filtered_dataset = Subset(dataset, valid_indices)


# # 2.2) Instead of filtering by ThresholdSampler, just use the full dataset:
# filtered_dataset = dataset
# valid_indices     = list(range(len(filtered_dataset)))   



# ---------------------------------------------------
# ——— Set up K-Fold ———

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# storage for fold results
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(valid_indices), start=1):
    print(f"\n==== Fold {fold}/{K} ====")


    # create subsets for this fold
    train_ds = Subset(filtered_dataset, train_idx)
    val_ds   = Subset(filtered_dataset, val_idx)


    ### average over the test set 
    sum_lm = torch.zeros(50, 3)
    for idx in train_ds.indices:      # train_ds is a torch.utils.data.Subset
        _, lm, _ = dataset[idx]       # dataset is your MeshLandmarkDataset
        sum_lm += lm                  # lm is a (50,3) tensor
    mean_lm = sum_lm / len(train_ds)
    # print(mean_lm.shape)


    train_ds_patches = PatchDataset(
    # train_ds_patches = PatchDataset_distance_based(
        base_ds    = train_ds,
        mean_lm    = mean_lm,
        patch_size = 1000,
        # patch_size = 1500,
        cache_dir  = "patch_cache_train"
    )

    val_ds_patches = PatchDataset(
    # val_ds_patches = PatchDataset_distance_based(
        base_ds    = val_ds,
        mean_lm    = mean_lm,
        patch_size = 1000,
        # patch_size = 1500,
        cache_dir  = "patch_cache_test"
    )


    train_loader_patches = DataLoader(train_ds_patches,
                            batch_size=16,
                            shuffle=False,      # shuffling here is OK
                            num_workers=4,
                            pin_memory=True,
                            )


    val_loader_patches = DataLoader(val_ds_patches,
                            batch_size=16,
                            shuffle=False,      # shuffling here is OK
                            num_workers=4,
                            pin_memory=True,
                            )





    print("Pre‐caching patches (Train_set)…")
    for i in tqdm(range(len(train_ds_patches)), desc="Caching patches"):
        _ = train_ds_patches[i]

    print("Pre‐caching patches (Validation Set)…")
    for i in tqdm(range(len(val_ds_patches)), desc="Caching patches"):
        _ = val_ds_patches[i]

    for data in train_loader_patches:
        X_patch_sample, y_landmark, X_sampled_sample = data
        break


    #---------------------------------------------------------------

    # ——— Set up Model ———
    from src.models.model import PALNET, PLNET_noatt, PALNET_topk, PALNET_2blk
    from src.models.loss import CombinedLoss, localizationLoss 
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_shape = X_patch_sample[0].shape
    output_shape = y_landmark[0].shape

    # Define model
    model = PALNET(input_shape, output_shape, seed=42).to(device, non_blocking=True)

    #### for ab study
    # model = PLNET_noatt(input_shape, output_shape, seed=42).to(device, non_blocking=True)
    # model = PALNET_topk(input_shape, output_shape, seed=42).to(device, non_blocking=True)
    # model = PALNET_2blk(input_shape, output_shape, seed=42).to(device, non_blocking=True)

    # Define loss function & optimizer
    criterion = CombinedLoss(alpha=0.6, beta=0.4)  
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    num_epochs = 5000  # Adjust as needed
    model.to(device, non_blocking=True)

    # Early Stopping Parameters
    patience = 30 
    best_val_loss = float('inf')
    epochs_no_improve = 0



    # ------------------------------------------------------------
    # ——— Training Loop ———

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        ## training loop
        for X_patches, y_landmarks, _ in train_loader_patches:

            X_patches = X_patches.to(device, non_blocking=True)
            y_landmarks = y_landmarks.to(device, non_blocking=True)

            optimizer.zero_grad()  # Zero gradients
            outputs = model(X_patches)  # Forward pass
            loss = criterion(outputs, y_landmarks)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item() * X_patches.size(0)
        train_loss = running_loss / len(train_ds)

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for X_val, y_val, _ in val_loader_patches:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item() * X_val.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        
        # Print Epoch Summary
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            bestmodel = model.state_dict()
            if epoch > 50:
                torch.save(model.state_dict(), "best_model_ref"+str(fold) +".pth")  # Save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break  # Stop training if no improvement for `patience` epochs
        
        print("best_val_loss: ", best_val_loss)

    # Load Best Model
    model.load_state_dict(torch.load("best_model_ref"+str(fold) +".pth"))
    print("Best model loaded!")
    # Save the model for this fold


    #---------------------------------------------------------------
    # ——— Evaluation Loop ———
    pred = []
    y_test = []
    X_test = []
    X_test_raw = []

    for X_patches, y_landmarks, _ in val_loader_patches:
        X_patches = X_patches.to(device, non_blocking=True)
        y_landmarks = y_landmarks.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(X_patches)
            pred.append(outputs.cpu().numpy())
            y_test.append(y_landmarks.cpu().numpy())
            X_test.append(_.cpu().numpy())
            X_test_raw.append(X_patches.cpu().numpy())

    pred = np.concatenate(pred, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    X_test_raw = np.concatenate(X_test_raw, axis=0)
    X_test_raw = X_test_raw.reshape(X_test_raw.shape[0], -1, 3)


    preds_np_closes = fix_prediction(X_test, pred.copy())
    preds_np_average = fix_precition_average(X_test, pred.copy(), k =12)

    # Compute distance error
    distance_error_closes  = compute_distance_error(y_test, preds_np_closes)
    distance_error_average = compute_distance_error(y_test, preds_np_average)

    err_closes  = compute_pointwise_error(preds_np_closes, y_test)
    err_average = compute_pointwise_error(preds_np_average, y_test)


    print("___________ Evaluation "+str(fold)+"___________")
    print("Pair-wise_closes:" , distance_error_closes.mean())
    print("Pair-wise_average:" , distance_error_average.mean())
    print()
    print("Point-wise_closes:" ,err_closes.mean())
    print("Point-wise_average:" ,err_average.mean())
    

    df_results[f"Fold_{fold}_Point-wise_closes"] = err_closes
    df_results[f"Fold_{fold}_Point-wise_average"] = err_average
    point_wise_closes.append(err_closes)
    point_wise_average.append(err_average)
    distance_wise_closes.append(distance_error_closes)
    distance_wise_average.append(distance_error_average)

    # Save results to CSV
    path_to_save = "results_"+str(fold) +".csv"
    df_results.to_csv(path_to_save, mode='a', header=False, index=False)


    import shutil
    shutil.rmtree("patch_cache_train")
    shutil.rmtree("patch_cache_test")



point_wise_closes = np.array(point_wise_closes)
point_wise_average = np.array(point_wise_average)
distance_wise_closes = np.array(distance_wise_closes)
distance_wise_average = np.array(distance_wise_average)


np.save("point_wise_closes.npy", point_wise_closes)
np.save("point_wise_average.npy", point_wise_average)
np.save("distance_wise_closes.npy", distance_wise_closes)
np.save("distance_wise_average.npy", distance_wise_average)

