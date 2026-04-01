import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import seaborn as sns
import cv2


# ══════════════════════════════════════════════════════════════
# Custom PyTorch Dataset for multi-label hemorrhage classification
# Parses the RSNA CSV format where each row is image_type (e.g.
# ID_abc_epidural) and maps it to a 6-element binary label vector:
# [epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any]
# ══════════════════════════════════════════════════════════════
class HemorrhageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, folder, transforms=None):
        
        df = pd.read_csv(csv_file)
        
        labels_temp = {}
        for _, row in df.iterrows():
            img_id = '_'.join(row['ID'].split('_')[:-1])
            label_type = row['ID'].split('_')[-1]
            
            if img_id not in labels_temp:
                labels_temp[img_id] = [0, 0, 0, 0, 0, 0]
            
            label_idx = {
                'epidural': 0,
                'intraparenchymal': 1,
                'intraventricular': 2,
                'subarachnoid': 3,
                'subdural': 4,
                'any': 5
            }
            
            labels_temp[img_id][label_idx[label_type]] = int(row['Label'])
        
        self.image_files = list(labels_temp.keys())
        self.labels = list(labels_temp.values())
        self.folder = folder
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_id = self.image_files[index]
        img_path = os.path.join(self.folder, f"{img_id}_frame0.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        label_list = self.labels[index]
        labels = torch.tensor(label_list, dtype=torch.float32)
        
        return image, labels

# ──────────────────────────────────────────────────────────────
# Random 80/20 train/validation split using shuffled indices
# ──────────────────────────────────────────────────────────────
def split_train_validation(dataset, train_ratio=0.8):
    length = len(dataset)
    indices = np.arange(length)
    np.random.shuffle(indices)
    
    split_point = int(train_ratio * length)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    print(f"Dataset original: {length} imagini")
    print(f"Train set: {len(train_indices)} imagini ({len(train_indices)/length*100:.1f}%)")
    print(f"Validation set: {len(val_indices)} imagini ({len(val_indices)/length*100:.1f}%)")
    
    return train_indices, val_indices


# ──────────────────────────────────────────────────────────────
# Analyze and visualize the class distribution across all 6
# hemorrhage types. Computes per-class counts, percentages,
# and imbalance ratio. Generates histogram subplots.
# ──────────────────────────────────────────────────────────────
def analyze_class_distribution(dataset, indices, dataset_name):
    
    categories = ['Epidural', 'Intraparenchymal', 'Intraventricular',
                  'Subarachnoid', 'Subdural', 'Any']
    
    all_labels = []
    for idx in indices:
        _, labels = dataset[idx]
        all_labels.append(labels.numpy())
    
    df_labels = pd.DataFrame(all_labels, columns=categories)
    
    print(df_labels.describe())
    
    counts = df_labels.sum()
    total_samples = len(indices)
    
    for cat in categories:
        count = counts[cat]
        percentage = (count / total_samples) * 100
        print(f"{cat:20s}: {int(count):5d} ({percentage:5.2f}%)")
    
    max_count = max(counts.values)
    min_count = min([c for c in counts.values if c > 0])
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Raport dezechilibru: {imbalance_ratio:.2f}:1\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(categories):
        sns.histplot(data=df_labels, x=cat, ax=axes[i], bins=2, 
                    kde=False, color='steelblue', edgecolor='black')
        
        count_0 = (df_labels[cat] == 0).sum()
        count_1 = (df_labels[cat] == 1).sum()
        
        axes[i].text(0, count_0, f'{count_0}\n({count_0/total_samples*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        axes[i].text(1, count_1, f'{count_1}\n({count_1/total_samples*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        axes[i].set_title(f'{cat}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Label (0=Absent, 1=Present)', fontsize=10)
        axes[i].set_ylabel('Numar Imagini', fontsize=10)
        axes[i].set_xticks([0, 1])
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Histograme Distributie Clase - {dataset_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename_hist = f'histograms_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename_hist, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return counts.values, imbalance_ratio

# ──────────────────────────────────────────────────────────────
# Display sample CT images for each hemorrhage type and compute
# the co-occurrence matrix (which types appear together)
# ──────────────────────────────────────────────────────────────
def visualize_hemorrhage_samples(dataset, indices, num_samples_per_class=10, dataset_name="Train"):
    
    categories = ['Epidural', 'Intraparenchymal', 'Intraventricular',
                  'Subarachnoid', 'Subdural']
    
    category_indices = {
        'Epidural': 0,
        'Intraparenchymal': 1,
        'Intraventricular': 2,
        'Subarachnoid': 3,
        'Subdural': 4
    }
    
    samples_per_category = {cat: [] for cat in categories}
    
    for idx in indices:
        _, labels = dataset[idx]
        labels_np = labels.numpy()
        
        for cat in categories:
            cat_idx = category_indices[cat]
            if labels_np[cat_idx] == 1 and len(samples_per_category[cat]) < num_samples_per_class:
                samples_per_category[cat].append(idx)
    
    for cat in categories:
        if len(samples_per_category[cat]) == 0:
            continue
        
        samples = samples_per_category[cat]
        n_samples = len(samples)
        
        n_cols = min(5, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{cat} - {dataset_name} ({n_samples} mostre)',
                    fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(samples):
            row = i // n_cols
            col = i % n_cols
            
            img_id = dataset.image_files[idx]
            img_path = os.path.join(dataset.folder, f"{img_id}_frame0.png")
            image = Image.open(img_path).convert('RGB')
            
            labels = dataset.labels[idx]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].axis('off')
            
            other_hemorrhages = []
            for other_cat, other_idx in category_indices.items():
                if other_cat != cat and labels[other_idx] == 1:
                    other_hemorrhages.append(other_cat[:3])
            
            title = f"#{i+1}"
            if other_hemorrhages:
                title += f"\n+{','.join(other_hemorrhages)}"
            
            axes[row, col].set_title(title, fontsize=9)
        
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        filename = f'samples_{cat.lower()}_{dataset_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    cooccurrence_matrix = np.zeros((len(categories), len(categories)))
    
    for idx in indices:
        labels = dataset.labels[idx]
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i < j:
                    if labels[category_indices[cat1]] == 1 and labels[category_indices[cat2]] == 1:
                        cooccurrence_matrix[i, j] += 1
    
    print(f"\nCo-occurenta {dataset_name}:")
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j and cooccurrence_matrix[i, j] > 0:
                print(f"{cat1:20s} + {cat2:20s}: {int(cooccurrence_matrix[i, j]):4d}")

# ──────────────────────────────────────────────────────────────
# Verify image integrity: check channels, dimensions, and pixel
# value distributions (min, max, mean, std) across the dataset
# ──────────────────────────────────────────────────────────────
def verify_dataset_integrity(dataset, indices, dataset_name="Train"):
    
    channels_list = []
    dimensions_list = []
    pixel_values_stats = {
        'min': [],
        'max': [],
        'mean': [],
        'std': []
    }
    
    for idx in indices:
        img_id = dataset.image_files[idx]
        img_path = os.path.join(dataset.folder, f"{img_id}_frame0.png")
        
        image = Image.open(img_path)
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            channels = 1
            dimensions = img_array.shape
        else:
            channels = img_array.shape[2]
            dimensions = img_array.shape[:2]
        
        channels_list.append(channels)
        dimensions_list.append(dimensions)
        
        pixel_values_stats['min'].append(img_array.min())
        pixel_values_stats['max'].append(img_array.max())
        pixel_values_stats['mean'].append(img_array.mean())
        pixel_values_stats['std'].append(img_array.std())
    
    unique_channels = set(channels_list)
    channels_count = {ch: channels_list.count(ch) for ch in unique_channels}
    
    print(f"\n{dataset_name} - Canale:")
    for ch, count in sorted(channels_count.items()):
        percentage = (count / len(indices)) * 100
        print(f"  {ch} canal(e): {count} ({percentage:.2f}%)")
    
    unique_dimensions = set(dimensions_list)
    
    print(f"\n{dataset_name} - Dimensiuni:")
    if len(unique_dimensions) == 1:
        dim = list(unique_dimensions)[0]
        print(f"  Uniforma: {dim[0]}x{dim[1]}")
    else:
        dims_count = {}
        for dim in dimensions_list:
            dim_str = f"{dim[0]}x{dim[1]}"
            dims_count[dim_str] = dims_count.get(dim_str, 0) + 1
        
        for dim_str, count in sorted(dims_count.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(indices)) * 100
            print(f"  {dim_str}: {count} ({percentage:.2f}%)")
    
    print(f"\n{dataset_name} - Valori pixeli:")
    print(f"  Range: [{min(pixel_values_stats['min']):.0f}, {max(pixel_values_stats['max']):.0f}]")
    print(f"  Mean: {np.mean(pixel_values_stats['mean']):.2f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(pixel_values_stats['min'], bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_title('Distributie Min')
    axes[0, 0].set_xlabel('Min')
    
    axes[0, 1].hist(pixel_values_stats['max'], bins=50, color='red', alpha=0.7)
    axes[0, 1].set_title('Distributie Max')
    axes[0, 1].set_xlabel('Max')
    
    axes[1, 0].hist(pixel_values_stats['mean'], bins=50, color='green', alpha=0.7)
    axes[1, 0].set_title('Distributie Mean')
    axes[1, 0].set_xlabel('Mean')
    
    axes[1, 1].hist(pixel_values_stats['std'], bins=50, color='orange', alpha=0.7)
    axes[1, 1].set_title('Distributie Std')
    axes[1, 1].set_xlabel('Std')
    
    plt.suptitle(f'Statistici Pixeli - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'pixel_stats_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ──────────────────────────────────────────────────────────────
# Compare image preprocessing techniques on sample CT scans:
# Ben Graham (HSV enhancement), CLAHE (adaptive histogram eq.),
# Sobel (edge detection), Gaussian Blur, and [0,1] normalization
# ──────────────────────────────────────────────────────────────
def apply_preprocessing_techniques(dataset, indices, num_samples=5, dataset_name="Train"):
    
    np.random.seed(42)
    sample_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
    
    def ben_graham_preprocessing(img):
        img_uint8 = (img).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        result = cv2.addWeighted(img_hsv, 4, cv2.GaussianBlur(img_hsv, (0,0), 512/10), -4, 128)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    def clahe_preprocessing(img):
        img_uint8 = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img_uint8).astype(np.float32)
    
    def sobel_edges(img):
        img_uint8 = img.astype(np.uint8)
        sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        return edges.astype(np.float32)
    
    def gaussian_blur(img):
        img_uint8 = img.astype(np.uint8)
        return cv2.GaussianBlur(img_uint8, (5, 5), 0).astype(np.float32)
    
    def normalize_image(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    techniques = {
        'Original': lambda img: img,
        'Ben Graham': ben_graham_preprocessing,
        'CLAHE': clahe_preprocessing,
        'Sobel Edges': sobel_edges,
        'Gaussian Blur': gaussian_blur,
        'Normalizare [0,1]': normalize_image
    }
    
    for sample_idx, idx in enumerate(sample_indices):
        img_id = dataset.image_files[idx]
        img_path = os.path.join(dataset.folder, f"{img_id}_frame0.png")
        
        original_img = Image.open(img_path).convert('L')
        img_array = np.array(original_img, dtype=np.float32)
        
        n_techniques = len(techniques)
        n_cols = 3
        n_rows = (n_techniques + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten()
        
        for i, (tech_name, tech_func) in enumerate(techniques.items()):
            processed_img = tech_func(img_array.copy())
            axes[i].imshow(processed_img, cmap='gray')
            axes[i].set_title(f'{tech_name}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        for i in range(n_techniques, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Tehnici Preprocesare - Sample {sample_idx+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'preprocessing_sample{sample_idx+1}_{dataset_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    
    from torchvision import transforms
    
    simple_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    train_dataset = HemorrhageDataset(
        csv_file='archive/subdataset_train.csv',
        folder='archive/rsna-intracranial-hemorrhage-detection-png/train_images',
        transforms=simple_transforms
    )
    
    test_dataset = HemorrhageDataset(
        csv_file='archive/subdataset_test.csv',
        folder='archive/rsna-intracranial-hemorrhage-detection-png/test_images',
        transforms=simple_transforms
    )
    
    train_indices, val_indices = split_train_validation(train_dataset, train_ratio=0.8)
    
    train_counts, train_ratio = analyze_class_distribution(train_dataset, train_indices, "Train")
    val_counts, val_ratio = analyze_class_distribution(train_dataset, val_indices, "Validation")
    test_indices_list = list(range(len(test_dataset)))
    test_counts, test_ratio = analyze_class_distribution(test_dataset, test_indices_list, "Test")
    
    visualize_hemorrhage_samples(train_dataset, train_indices, num_samples_per_class=10, dataset_name="Train")
    visualize_hemorrhage_samples(test_dataset, test_indices_list, 10, "Test")
    
    verify_dataset_integrity(train_dataset, train_indices, dataset_name="Train")
    verify_dataset_integrity(test_dataset, test_indices_list, dataset_name="Test")
    
    apply_preprocessing_techniques(train_dataset, train_indices, num_samples=5, dataset_name="Train")
