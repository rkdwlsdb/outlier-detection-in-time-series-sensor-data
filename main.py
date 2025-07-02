import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import defaultdict

# 1. 파일 그룹 정의
env_groups = {
    "env1": ["B0005.csv", "B0006.csv", "B0007.csv", "B0018.csv"],
    "env2": ["B0033.csv", "B0034.csv", "B0036.csv"],
    "env3": ["B0038.csv", "B0039.csv", "B0040.csv"],
}
base_path = r"data"
file_paths = {fn: os.path.join(base_path, fn) for group in env_groups.values() for fn in group}
all_features = ["Voltage_measured", "Current_measured", "Temperature_measured"]
outputs_base = 'outputs'

# 매직 넘버들을 상수로 정의
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
QUANTILE_THRESHOLD = 0.95

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-8)

def extract_and_pad_multivariate(files, feature_list):
    all_dfs = [pd.read_csv(file_paths[f]) for f in files]
    max_len = 0
    for df in all_dfs:
        for _, group in df.groupby("cycle_idx"):
            max_len = max(max_len, len(group))
    all_vectors = []
    all_labels = []
    for file_name, df in zip(files, all_dfs):
        grouped = df.groupby("cycle_idx")
        for cycle_id, group in grouped:
            vector = []
            skip = False
            for feat in feature_list:
                seq = group[feat].values
                if len(seq) < 2:
                    skip = True
                    break
                seq = normalize(seq)
                if len(seq) < max_len:
                    seq = np.pad(seq, (0, max_len - len(seq)))
                else:
                    seq = seq[:max_len]
                vector.extend(seq)
            if skip:
                print(f"Skip {file_name} cycle {cycle_id} (length too short)")
                continue
            all_vectors.append(vector)
            all_labels.append((file_name, cycle_id))
    all_vectors = np.array(all_vectors)
    masks = (all_vectors != 0).astype(float)
    return all_vectors, masks, all_labels, max_len

def compare_thresholds(recon_error):
    thr_dict = {}
    thr_dict['STD'] = recon_error.mean() + 3 * recon_error.std()
    q1, q3 = np.percentile(recon_error, [25, 75])
    iqr = q3 - q1
    thr_dict['IQR'] = q3 + 1.5 * iqr
    thr_dict['Quantile'] = np.quantile(recon_error, 0.95)
    return thr_dict

def compare_reconstruction_error(
    recon_error_before, threshold_dict,
    recon_error_after, threshold_dict_after,
    save_dir=None, fname=None
):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.hist(recon_error_before, bins=50, alpha=0.6, color='gray', label='Before')
    for key, thr in threshold_dict.items():
        plt.axvline(thr, linestyle='--', label=f"{key} threshold")
    plt.title("Before Outlier Removal")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(recon_error_after, bins=50, alpha=0.6, color='blue', label='After')
    for key, thr in threshold_dict_after.items():
        plt.axvline(thr, linestyle='--', label=f"{key} threshold")
    plt.title("After Outlier Removal")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    # *** 자동 저장 ***
    if save_dir is None:
        save_dir = "./auto_plots"
    os.makedirs(save_dir, exist_ok=True)
    if fname is None:
        fname = "compare_recon_error.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

class MultiAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def masked_mse_loss(input, target, mask):
    squared_error = (input - target) ** 2
    masked_se = squared_error * mask
    return masked_se.sum() / mask.sum()

def evaluate_performance(recon_error, tag=""):
    print(f"[{tag}] Reconstruction Error Mean: {np.mean(recon_error):.6f}, "
          f"Std: {np.std(recon_error):.6f}")
    print(f"[{tag}] Min: {np.min(recon_error):.6f}, Max: {np.max(recon_error):.6f}")

def train_autoencoder(X, masks, num_epochs=50, batch_size=32, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    mask_tensor = torch.tensor(masks, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, mask_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MultiAutoencoder(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = masked_mse_loss(x_hat, x, mask)
            loss.backward()
            optimizer.step()
    return model

def detect_outliers(model, X, masks, labels, method='quantile', threshold_std=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(masks, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        X_hat = model(X_tensor).cpu().numpy()
    recon_error = np.sum(((X - X_hat) ** 2) * masks, axis=1) / np.sum(masks, axis=1)
    if method == 'std':
        threshold = recon_error.mean() + threshold_std * recon_error.std()
    elif method == 'iqr':
        q1, q3 = np.percentile(recon_error, [25, 75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
    elif method == 'quantile':
        threshold = np.quantile(recon_error, 0.95)
    else:
        raise ValueError("지원하지 않는 method")
    outlier_indices = [i for i, err in enumerate(recon_error) if err > threshold]
    outlier_labels = [labels[i] for i in outlier_indices]
    return outlier_labels, outlier_indices, recon_error, threshold

def plot_all_cycles_with_outliers_by_file(
    recon_error, labels_used, outlier_indices, file_names, env_name, save_dir_base=None
):
    for file_name in file_names:
        file_indices = [i for i, x in enumerate(labels_used) if x[0] == file_name]
        if not file_indices:
            continue
        cycle_indices = [labels_used[i][1] for i in file_indices]
        recon_error_file = [recon_error[i] for i in file_indices]
        outlier_indices_file = [idx for idx in range(len(file_indices)) 
                                if file_indices[idx] in outlier_indices]
        plt.figure(figsize=(12, 5))
        plt.scatter(cycle_indices, recon_error_file, color='gray', s=30, label='Normal')
        if outlier_indices_file:
            plt.scatter(
                [cycle_indices[i] for i in outlier_indices_file],
                [recon_error_file[i] for i in outlier_indices_file],
                color='blue', s=50, label='Outliers'
            )
        plt.xlabel("Cycle Index")
        plt.ylabel("Reconstruction Error")
        plt.title(f"{env_name} - {file_name} - Multivariate Outlier Visualization")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # 자동저장
        if save_dir_base:
            os.makedirs(save_dir_base, exist_ok=True)
            fname = f"{file_name.replace('.csv','')}_outlier_scatter.png"
            plt.savefig(os.path.join(save_dir_base, fname))
        plt.close()

def plot_cycles_all(file_path, feature, cleaned_cycles=None, removed_cycles=None, save_dir=None):
    import matplotlib.lines as mlines
    df = pd.read_csv(file_path)
    cycles = sorted(df['cycle_idx'].unique())
    max_len = max(len(df[df['cycle_idx'] == c][feature]) for c in cycles)

    plt.figure(figsize=(18, 6))
    for i, c in enumerate(cycles):
        sub = df[df['cycle_idx'] == c][feature]
        plt.plot(np.arange(len(sub)), sub, color='gray', alpha=0.15, linewidth=1)
    if cleaned_cycles is not None:
        for i, c in enumerate(cleaned_cycles):
            sub = df[df['cycle_idx'] == c][feature]
            plt.plot(np.arange(len(sub)), sub, color='blue', alpha=0.5, linewidth=1.2)
    if removed_cycles is not None:
        for i, c in enumerate(removed_cycles):
            sub = df[df['cycle_idx'] == c][feature]
            plt.plot(np.arange(len(sub)), sub, color='red', alpha=0.9, linewidth=2)
    legend_elements = [
        mlines.Line2D([], [], color='gray', alpha=0.3, label='Original'),
        mlines.Line2D([], [], color='blue', label='Cleaned'),
        mlines.Line2D([], [], color='red', label='Removed Outlier'),
    ]
    plt.legend(handles=legend_elements)
    plt.title(f"{os.path.basename(file_path)} - {feature} (Cycle-wise overlapped)")
    plt.xlabel('Tick (time)')
    plt.ylabel(feature)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{os.path.basename(file_path).replace('.csv','')}_{feature}_cyclewise_overlapped.png"
        plt.savefig(os.path.join(save_dir, fname))
    plt.close()


# main
set_seed(42)
for env_name, files in env_groups.items():
    print(f"\n=== 실험 환경: {env_name} (Multivariate) ===")
    # 1. 데이터 준비 (모든 feature 합침)
    X, masks, labels_used, max_len = extract_and_pad_multivariate(files, all_features)
    print(f"Total cycles: {len(X)} | Vector dim: {X.shape[1]}")

    # 2. Autoencoder 학습 및 성능 평가 
    model = train_autoencoder(X, masks)

    # 복원오차 계산
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_hat = model(X_tensor).cpu().numpy()
    recon_error = np.sum(((X - X_hat) ** 2) * masks, axis=1) / np.sum(masks, axis=1)

    # 오차 기준 평가
    evaluate_performance(recon_error, tag="Before")
    thr_dict = compare_thresholds(recon_error)
    print("Thresholds:", thr_dict)

    # 3. 이상치 탐지
    outlier_labels, outlier_indices, recon_error, thr = detect_outliers(
        model, X, masks, labels_used, method='quantile'
    )
    print(f"[{env_name}] Detected {len(outlier_indices)} outliers (Quantile 95%)")

    # 4. 이상치 저장
    outlier_rows = []

    for idx, (fname, cyc) in enumerate(labels_used):
        if idx in outlier_indices:
            outlier_rows.append([fname, cyc, recon_error[idx]])

    # 환경 전체 outlier 목록만 저장 (env1_outliers.csv)
    if outlier_rows:
        env_outlier_path = os.path.join(outputs_base, env_name, f"{env_name}_outliers.csv")
        os.makedirs(os.path.dirname(env_outlier_path), exist_ok=True)
        with open(env_outlier_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "cycle_idx", "recon_error"])
            for fname, cyc, err in outlier_rows:
                writer.writerow([fname, cyc, err])

    # 5. 시각화
    plot_all_cycles_with_outliers_by_file(
        recon_error, labels_used, outlier_indices, files, env_name,
        save_dir_base=os.path.join("outputs", env_name, "outlier_scatter")
    )

    # 6. 이상치 제거 및 재학습 + 성능평가 + 히스토그램
    filtered_indices = [i for i in range(len(X)) if i not in set(outlier_indices)]
    X_clean = X[filtered_indices]
    masks_clean = masks[filtered_indices]
    labels_clean = [labels_used[i] for i in filtered_indices]
    model_clean = train_autoencoder(X_clean, masks_clean)
    with torch.no_grad():
        X_clean_tensor = torch.tensor(X_clean, dtype=torch.float32).to(device)
        X_hat_clean = model_clean(X_clean_tensor).cpu().numpy()
    recon_error_clean = np.sum(((X_clean - X_hat_clean) ** 2) * masks_clean, axis=1) / np.sum(masks_clean, axis=1)
    evaluate_performance(recon_error_clean, tag="After")
    thr_dict_clean = compare_thresholds(recon_error_clean)
    print("Thresholds (After):", thr_dict_clean)
    compare_reconstruction_error(
        recon_error, thr_dict, recon_error_clean, thr_dict_clean,
        save_dir=os.path.join("outputs", env_name, "recon_error_compare"),
        fname=f"{env_name}_compare_recon_error.png"
    )

    # 7. 이상치 제거 데이터 저장/시각화
    outlier_set = set(outlier_indices)
    labels_filtered = [labels_used[i] for i in filtered_indices]
    file_cycle_dict = {fn: set() for fn in files}
    for fname, cyc in labels_filtered:
        file_cycle_dict[fname].add(cyc)
    for file_name in files:
        raw_path = os.path.join(base_path, file_name)
        raw_df = pd.read_csv(raw_path)
        keep_cycles = file_cycle_dict[file_name]
        cleaned_raw_df = raw_df[raw_df["cycle_idx"].isin(list(keep_cycles))].copy()
        out_merge_dir = os.path.join("outputs", env_name, file_name.replace('.csv', ''))
        os.makedirs(out_merge_dir, exist_ok=True)
        final_save_path = os.path.join(out_merge_dir, f"{file_name.replace('.csv','')}_cleaned_final.csv")
        cleaned_raw_df.to_csv(final_save_path, index=False)
        # print(f"[{env_name}] {file_name} multivariate cleaned saved: {final_save_path}")

# 전체 비교 그래프 
for env_name, files in env_groups.items():
    for file_name in files:
        raw_path = os.path.join(base_path, file_name)
        cleaned_path = os.path.join("outputs", env_name, file_name.replace('.csv',''), f"{file_name.replace('.csv','')}_cleaned_final.csv")
        if not (os.path.exists(raw_path) and os.path.exists(cleaned_path)):
            print(f"Skip {file_name} (missing file)")
            continue
        raw_df = pd.read_csv(raw_path)
        cleaned_df = pd.read_csv(cleaned_path)
        all_cycles = set(raw_df['cycle_idx'].unique())
        cleaned_cycles = set(cleaned_df['cycle_idx'].unique())
        removed_cycles = sorted(all_cycles - cleaned_cycles)
        save_dir = os.path.join("outputs", env_name, "cyclewise_compare")
        os.makedirs(save_dir, exist_ok=True)

        for feature in all_features:
            if feature not in raw_df.columns:
                continue
            # 파일명에 배터리명을 붙임
            fname = f"{file_name.replace('.csv','')}_{feature}_cyclewise_overlapped.png"
            plot_cycles_all(
                raw_path, feature,
                cleaned_cycles=cleaned_cycles,
                removed_cycles=removed_cycles,
                save_dir=save_dir  # plot_cycles_all에서 fname으로 저장
            )

