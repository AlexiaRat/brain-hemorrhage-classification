import os
import sys
import time
import copy
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")



from google.colab import drive
drive.mount('/content/drive', force_remount=False)




ZIP_PATH = '/content/drive/MyDrive/archive.zip'
EXTRACT_TO = '/content/archive'

BASE_PATH = EXTRACT_TO

def find_file(base, filename):
    for root, dirs, files in os.walk(base):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_folder(base, foldername):
    for root, dirs, files in os.walk(base):
        if foldername in dirs:
            return os.path.join(root, foldername)
    return None

csv_train = find_file(BASE_PATH, 'subdataset_train.csv') or 'archive/subdataset_train.csv'
csv_test = find_file(BASE_PATH, 'subdataset_test.csv') or 'archive/subdataset_test.csv'

folder_train = find_folder(BASE_PATH, 'train_256') or find_folder(BASE_PATH, 'train_images') or 'archive/train_256'
folder_test = find_folder(BASE_PATH, 'test_256') or find_folder(BASE_PATH, 'test_images') or 'archive/test_256'

print(f"CSV Train: {csv_train}")
print(f"CSV Test: {csv_test}")
print(f"Folder Train: {folder_train}")
print(f"Folder Test: {folder_test}")

OUTPUT_DIR = '/content/outputs'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix, roc_curve,
                            precision_recall_curve, classification_report)
from sklearn.model_selection import StratifiedKFold

try:
    from monai.transforms import (
        Compose, RandRotate, RandFlip, RandZoom, RandGaussianNoise,
        RandAdjustContrast, RandGaussianSmooth
    )
except ImportError:
    os.system('pip install -q monai')
    from monai.transforms import (
        Compose, RandRotate, RandFlip, RandZoom, RandGaussianNoise,
        RandAdjustContrast, RandGaussianSmooth
    )

DEVICE = torch.device('cuda')


# ══════════════════════════════════════════════════════════════
# Training configuration — centralized hyperparameters
# ══════════════════════════════════════════════════════════════
class Config:
    CSV_TRAIN = csv_train
    CSV_TEST = csv_test
    FOLDER_TRAIN = folder_train
    FOLDER_TEST = folder_test
    OUTPUT_DIR = OUTPUT_DIR
    PLOTS_DIR = PLOTS_DIR
    
    IMAGE_SIZE = 224
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.01
    K_FOLDS = 5
    PATIENCE = 3
    
    DEVICE = DEVICE
    SEED = 42
    
    CLASS_NAMES = ['Epidural', 'Intraparenchymal', 'Intraventricular', 
                   'Subarachnoid', 'Subdural', 'Any']

config = Config()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config.SEED)


class HemorrhageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, folder, transform=None, monai_transform=None):
        df = pd.read_csv(csv_file)
        labels_temp = {}
        
        for _, row in df.iterrows():
            img_id = '_'.join(row['ID'].split('_')[:-1])
            label_type = row['ID'].split('_')[-1]
            
            if img_id not in labels_temp:
                labels_temp[img_id] = [0, 0, 0, 0, 0, 0]
            
            label_idx = {'epidural': 0, 'intraparenchymal': 1, 
                        'intraventricular': 2, 'subarachnoid': 3, 
                        'subdural': 4, 'any': 5}
            labels_temp[img_id][label_idx[label_type]] = int(row['Label'])
        
        available = set(os.listdir(folder))
        available_ids = {f.replace('_frame0.png', '') for f in available 
                        if f.endswith('_frame0.png')}
        valid_ids = set(labels_temp.keys()) & available_ids
        
        self.image_files = [i for i in labels_temp.keys() if i in valid_ids]
        self.labels = [labels_temp[i] for i in self.image_files]
        self.folder = folder
        self.transform = transform
        self.monai_transform = monai_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, f"{self.image_files[idx]}_frame0.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.monai_transform and torch.is_tensor(image):
            image = self.monai_transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None, monai_transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.monai_transform = monai_transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = os.path.join(self.dataset.folder,
                               f"{self.dataset.image_files[real_idx]}_frame0.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.monai_transform and torch.is_tensor(image):
            image = self.monai_transform(image)
        
        label = torch.tensor(self.dataset.labels[real_idx], dtype=torch.float32)
        return image, label
    
    def get_labels(self):
        return [self.dataset.labels[i] for i in self.indices]


def get_base_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_monai_augment_set1():
    return Compose([
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate(range_x=0.1, prob=0.5),
    ])

def get_monai_augment_set2():
    return Compose([
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate(range_x=0.2, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        RandAdjustContrast(gamma=(0.9, 1.1), prob=0.3),
    ])

def get_monai_augment_set3():
    return Compose([
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate(range_x=0.3, prob=0.5),
        RandZoom(min_zoom=0.8, max_zoom=1.2, prob=0.5),
        RandAdjustContrast(gamma=(0.8, 1.2), prob=0.5),
        RandGaussianNoise(mean=0.0, std=0.05, prob=0.3),
    ])


from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ══════════════════════════════════════════════════════════════
# EfficientNet-V2-S with custom multi-label classifier head
# Backbone: pretrained ImageNet weights (optionally frozen)
# Head: Dropout → Linear(1280,512) → ReLU → Dropout → Linear(512,6)
# ══════════════════════════════════════════════════════════════
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5, freeze_backbone=False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# Focal Loss — reduces the contribution of easy examples, focusing
# training on hard-to-classify samples. Useful for imbalanced datasets.
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def calculate_class_weights(labels):
    labels_array = np.array(labels)
    pos = labels_array.sum(axis=0)
    neg = len(labels_array) - pos
    weights = neg / (pos + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def get_oversampling_weights(labels):
    labels_array = np.array(labels)
    class_counts = labels_array.sum(axis=0) + 1e-6
    class_weights = 1.0 / class_counts
    
    sample_weights = []
    for label in labels_array:
        weight = np.sum(label * class_weights)
        if weight == 0:
            weight = 1.0
        sample_weights.append(weight)
    
    return torch.tensor(sample_weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Single training epoch with optional mixed-precision (FP16)
# Returns average loss and exact-match accuracy
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, leave=False, desc="  Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append((probs > 0.5).astype(int))
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    accuracy = (all_preds == all_labels).all(axis=1).mean()
    
    return total_loss / len(loader), accuracy


# ──────────────────────────────────────────────────────────────
# Model evaluation — computes exact-match accuracy, Hamming score,
# precision, recall, F1, AUC (macro + per-class), and returns
# predictions and probabilities for downstream analysis
# ──────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="  Eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_preds.append((probs > threshold).astype(int))
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    hamming = 1 - np.mean(all_preds != all_labels)
    
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
        per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
    except:
        auc = 0.0
        per_class_auc = [0.0] * 6
    
    return {
        'loss': total_loss / len(loader),
        'exact_accuracy': exact_match,
        'hamming_score': hamming,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_auc': per_class_auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


# Early stopping — saves best model and stops training when
# validation loss doesn't improve for `patience` consecutive epochs
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, fold, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Fold {fold} - Evolutia Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'Fold {fold} - Evolutia Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fold{fold}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(labels, predictions, class_names, fold, plots_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, class_names)):
        cm = confusion_matrix(labels[:, i], predictions[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Absent', 'Present'],
                   yticklabels=['Absent', 'Present'])
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.suptitle(f'Fold {fold} - Matrice de Confuzie per Clasa', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fold{fold}_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(labels, probabilities, class_names, fold, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(labels[:, i], probabilities[:, i])
            auc_score = roc_auc_score(labels[:, i], probabilities[:, i])
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                   label=f'{name} (AUC = {auc_score:.3f})')
        except:
            pass
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Fold {fold} - ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fold{fold}_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, fold, plots_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics['per_class_precision'], width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, metrics['per_class_recall'], width, label='Recall', color='darkorange')
    bars3 = ax.bar(x + width, metrics['per_class_f1'], width, label='F1-Score', color='green')
    
    ax.set_xlabel('Clasa', fontsize=12)
    ax.set_ylabel('Scor', fontsize=12)
    ax.set_title(f'Fold {fold} - Metrici per Clasa', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/fold{fold}_per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_folds_comparison(all_fold_results, plots_dir, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    folds = range(1, len(all_fold_results) + 1)
    
    accuracies = [r['exact_accuracy'] for r in all_fold_results]
    hammings = [r['hamming_score'] for r in all_fold_results]
    f1s = [r['f1'] for r in all_fold_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='Exact Match Acc', color='steelblue')
    axes[0].bar(x + width/2, hammings, width, label='Hamming Score', color='darkorange')
    
    axes[0].axhline(y=0.4, color='red', linestyle='--', linewidth=2, label='Prag Accuracy (40%)')
    axes[0].axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Prag Hamming (70%)')
    
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('Scor', fontsize=12)
    axes[0].set_title(f'Comparatie Accuracy/Hamming intre Fold-uri{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Fold {f}' for f in folds])
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    bars3 = axes[1].bar(folds, f1s, color='green', alpha=0.7)
    axes[1].axhline(y=np.mean(f1s), color='red', linestyle='--', linewidth=2, 
                   label=f'Media F1 = {np.mean(f1s):.3f}')
    
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title(f'F1-Score per Fold{title_suffix}', fontsize=14, fontweight='bold')
    axes[1].set_xticks(folds)
    axes[1].set_xticklabels([f'Fold {f}' for f in folds])
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, f1s):
        axes[1].annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/folds_comparison{title_suffix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_experiments_comparison(all_experiments, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    names = list(all_experiments.keys())
    accuracies = [all_experiments[n]['test']['exact_accuracy']['mean'] for n in names]
    hammings = [all_experiments[n]['test']['hamming_score']['mean'] for n in names]
    f1s = [all_experiments[n]['test']['f1']['mean'] for n in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    axes[0].bar(x - width, accuracies, width, label='Exact Match Acc', color='steelblue')
    axes[0].bar(x, hammings, width, label='Hamming Score', color='darkorange')
    axes[0].bar(x + width, f1s, width, label='F1-Score', color='green')
    
    axes[0].axhline(y=0.4, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axhline(y=0.7, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    axes[0].set_xlabel('Experiment', fontsize=12)
    axes[0].set_ylabel('Scor', fontsize=12)
    axes[0].set_title('Comparatie Metrici intre Experimente', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    sorted_data = sorted(zip(names, f1s), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_f1s = zip(*sorted_data)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = axes[1].barh(range(len(sorted_names)), sorted_f1s, color=colors)
    
    axes[1].set_yticks(range(len(sorted_names)))
    axes[1].set_yticklabels(sorted_names, fontsize=10)
    axes[1].set_xlabel('F1-Score', fontsize=12)
    axes[1].set_title('Ranking Experimente (dupa F1)', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, sorted_f1s):
        axes[1].annotate(f'{val:.3f}',
                        xy=(val, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/experiments_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_results(ablation_results, plots_dir):
    df = pd.DataFrame(ablation_results)
    
    categories = df['category'].unique()
    n_cats = len(categories)
    
    fig, axes = plt.subplots(1, n_cats, figsize=(6*n_cats, 5))
    if n_cats == 1:
        axes = [axes]
    
    for ax, cat in zip(axes, categories):
        cat_data = df[df['category'] == cat]
        
        x = np.arange(len(cat_data))
        width = 0.25
        
        ax.bar(x - width, cat_data['accuracy'], width, label='Accuracy', color='steelblue')
        ax.bar(x, cat_data['hamming'], width, label='Hamming', color='darkorange')
        ax.bar(x + width, cat_data['f1'], width, label='F1', color='green')
        
        ax.set_xlabel('Configuratie', fontsize=11)
        ax.set_ylabel('Scor', fontsize=11)
        ax.set_title(f'Ablation: {cat}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_data['config'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('ABLATION STUDY - Rezultate', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_results_table(all_experiments, plots_dir):
    rows = []
    for name, results in all_experiments.items():
        if 'test' in results:
            rows.append({
                'Experiment': name,
                'Accuracy (%)': f"{results['test']['exact_accuracy']['mean']*100:.1f} +/- {results['test']['exact_accuracy']['std']*100:.1f}",
                'Hamming (%)': f"{results['test']['hamming_score']['mean']*100:.1f} +/- {results['test']['hamming_score']['std']*100:.1f}",
                'Precision': f"{results['test']['precision']['mean']:.3f} +/- {results['test']['precision']['std']:.3f}",
                'Recall': f"{results['test']['recall']['mean']:.3f} +/- {results['test']['recall']['std']:.3f}",
                'F1-Score': f"{results['test']['f1']['mean']:.3f} +/- {results['test']['f1']['std']:.3f}",
                'AUC': f"{results['test']['auc']['mean']:.3f} +/- {results['test']['auc']['std']:.3f}",
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(f'{plots_dir}/results_table.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(16, len(rows)*0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['lightsteelblue']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Tabel Comparativ - Toate Experimentele', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{plots_dir}/results_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return df


# ══════════════════════════════════════════════════════════════
# K-Fold cross-validation training loop
# Supports: weighted loss, oversampling, MONAI augmentation,
# early stopping, LR scheduling, and mixed precision training.
# Returns aggregated metrics (mean ± std) across all folds.
# ══════════════════════════════════════════════════════════════
def run_kfold_training(config, dataset, test_loader, use_weights=True, 
                       use_oversampling=False, monai_augment=None, 
                       use_early_stopping=True, use_scheduler=True,
                       description="Baseline", save_plots=True):
    

    
    stratify = np.array([l[5] for l in dataset.labels])
    kfold = StratifiedKFold(config.K_FOLDS, shuffle=True, random_state=config.SEED)
    
    all_fold_results = []
    all_test_results = []
    worst_fold_idx = 0
    worst_fold_f1 = float('inf')
    
    scaler = torch.cuda.amp.GradScaler()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(dataset)), stratify)):
    
        print(f"FOLD {fold+1}/{config.K_FOLDS}")
     
        
        train_transform = get_base_transform(config.IMAGE_SIZE)
        val_transform = get_val_transform(config.IMAGE_SIZE)
        
        train_sub = SubsetWithTransform(dataset, train_idx, train_transform, monai_augment)
        val_sub = SubsetWithTransform(dataset, val_idx, val_transform, None)
        
        if use_oversampling:
            train_labels = train_sub.get_labels()
            sample_weights = get_oversampling_weights(train_labels)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            train_loader = DataLoader(train_sub, config.BATCH_SIZE, sampler=sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_sub, config.BATCH_SIZE, shuffle=True, num_workers=2)
        
        val_loader = DataLoader(val_sub, config.BATCH_SIZE, num_workers=2)
        
        model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
        
        if use_weights:
            weights = calculate_class_weights(dataset.labels).to(config.DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE, 
                               weight_decay=config.WEIGHT_DECAY)
        
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=2, factor=0.5
            )
        
        if use_early_stopping:
            early_stopping = EarlyStopping(patience=config.PATIENCE)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_model_state = None
        best_val_loss = float('inf')
        
        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                                optimizer, config.DEVICE, scaler)
            val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
            
            train_losses.append(train_loss)
            val_losses.append(val_metrics['loss'])
            train_accs.append(train_acc)
            val_accs.append(val_metrics['exact_accuracy'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = copy.deepcopy(model.state_dict())
            
            if use_scheduler:
                scheduler.step(val_metrics['loss'])
            
            if use_early_stopping:
                if early_stopping(val_metrics['loss'], model):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch+1) % 3 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}: Loss={val_metrics['loss']:.4f}, "
                      f"Acc={val_metrics['exact_accuracy']*100:.1f}%, "
                      f"Hamming={val_metrics['hamming_score']*100:.1f}%")
        
        if save_plots:
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                               fold+1, config.PLOTS_DIR)
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        val_final = evaluate(model, val_loader, criterion, config.DEVICE)
        test_final = evaluate(model, test_loader, criterion, config.DEVICE)
        
        if save_plots:
            plot_confusion_matrices(test_final['labels'], test_final['predictions'],
                                  config.CLASS_NAMES, fold+1, config.PLOTS_DIR)
            plot_roc_curves(test_final['labels'], test_final['probabilities'],
                          config.CLASS_NAMES, fold+1, config.PLOTS_DIR)
            plot_per_class_metrics(test_final, config.CLASS_NAMES, fold+1, config.PLOTS_DIR)
        
        all_fold_results.append(val_final)
        all_test_results.append(test_final)
        
        if val_final['f1'] < worst_fold_f1:
            worst_fold_f1 = val_final['f1']
            worst_fold_idx = fold
        
        print(f"\n  Fold {fold+1} Results:")
        print(f"Val  - Acc: {val_final['exact_accuracy']*100:.1f}%, "
              f"Hamming: {val_final['hamming_score']*100:.1f}%, F1: {val_final['f1']:.4f}")
        print(f"Test - Acc: {test_final['exact_accuracy']*100:.1f}%, "
              f"Hamming: {test_final['hamming_score']*100:.1f}%, F1: {test_final['f1']:.4f}")
        
        torch.save(best_model_state, f'{config.OUTPUT_DIR}/fold{fold+1}_model.pth')
    
    if save_plots:
        plot_folds_comparison(all_test_results, config.PLOTS_DIR, f" - {description}")
    

    metrics_names = ['exact_accuracy', 'hamming_score', 'precision', 'recall', 'f1', 'auc']
    
    results_summary = {
        'description': description,
        'val': {},
        'test': {},
        'worst_fold': worst_fold_idx + 1
    }
    
    print(f"\n{'Metric':<20} {'Validation':<25} {'Test':<25}")
    
    for metric in metrics_names:
        val_vals = [r[metric] for r in all_fold_results]
        test_vals = [r[metric] for r in all_test_results]
        
        val_mean, val_std = np.mean(val_vals), np.std(val_vals)
        test_mean, test_std = np.mean(test_vals), np.std(test_vals)
        
        results_summary['val'][metric] = {'mean': float(val_mean), 'std': float(val_std)}
        results_summary['test'][metric] = {'mean': float(test_mean), 'std': float(test_std)}
        
        print(f"{metric:<20} {val_mean:.4f} +/- {val_std:.4f}      "
              f"{test_mean:.4f} +/- {test_std:.4f}")
    
    print(f"\nWorst fold: {worst_fold_idx + 1}")
    
    return results_summary, worst_fold_idx, all_fold_results, all_test_results


# ══════════════════════════════════════════════════════════════
# Ablation study — tests different loss functions (BCE, Weighted BCE,
# Focal Loss γ=2 and γ=3), dropout rates, and learning rates
# on the worst-performing fold to identify improvement opportunities
# ══════════════════════════════════════════════════════════════
def run_ablation_study(config, dataset, test_loader, worst_fold_idx):
    
    
    stratify = np.array([l[5] for l in dataset.labels])
    kfold = StratifiedKFold(config.K_FOLDS, shuffle=True, random_state=config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(dataset)), stratify)):
        if fold == worst_fold_idx:
            break
    
    train_transform = get_base_transform(config.IMAGE_SIZE)
    val_transform = get_val_transform(config.IMAGE_SIZE)
    
    train_sub = SubsetWithTransform(dataset, train_idx, train_transform)
    val_sub = SubsetWithTransform(dataset, val_idx, val_transform)
    
    ablation_results = []
    
    
    loss_configs = [
        ('BCE', nn.BCEWithLogitsLoss()),
        ('BCE+Weights', None),
        ('Focal(g=2)', FocalLoss(alpha=0.25, gamma=2.0)),
        ('Focal(g=3)', FocalLoss(alpha=0.5, gamma=3.0)),
    ]
    
    for loss_name, criterion in loss_configs:
        print(f"\n  Testing: {loss_name}")
        
        train_loader = DataLoader(train_sub, config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, config.BATCH_SIZE, num_workers=2)
        
        model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
        
        if criterion is None:
            weights = calculate_class_weights(dataset.labels).to(config.DEVICE)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        
        optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
        
        for epoch in range(5):
            train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        metrics = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), config.DEVICE)
        
        ablation_results.append({
            'category': 'Loss Function',
            'config': loss_name,
            'accuracy': metrics['exact_accuracy'],
            'hamming': metrics['hamming_score'],
            'f1': metrics['f1']
        })
        
        print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, "
              f"Hamming: {metrics['hamming_score']*100:.1f}%, F1: {metrics['f1']:.4f}")
    
    
    
    optimizer_configs = [
        ('Adam', lambda p: optim.Adam(p, lr=config.LEARNING_RATE)),
        ('AdamW', lambda p: optim.AdamW(p, lr=config.LEARNING_RATE, weight_decay=0.01)),
        ('SGD', lambda p: optim.SGD(p, lr=0.01, momentum=0.9)),
        ('RMSprop', lambda p: optim.RMSprop(p, lr=config.LEARNING_RATE)),
    ]
    
    for opt_name, opt_fn in optimizer_configs:
        print(f"\n  Testing: {opt_name}")
        
        train_loader = DataLoader(train_sub, config.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, config.BATCH_SIZE, num_workers=2)
        
        model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = opt_fn(model.parameters())
        
        for epoch in range(5):
            train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        
        ablation_results.append({
            'category': 'Optimizer',
            'config': opt_name,
            'accuracy': metrics['exact_accuracy'],
            'hamming': metrics['hamming_score'],
            'f1': metrics['f1']
        })
        
        print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, "
              f"Hamming: {metrics['hamming_score']*100:.1f}%, F1: {metrics['f1']:.4f}")
    
    
    
    for bs in [16, 32, 64]:
        print(f"\n  Testing: Batch Size = {bs}")
        
        train_loader = DataLoader(train_sub, bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, bs, num_workers=2)
        
        model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
        
        start_time = time.time()
        for epoch in range(5):
            train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_time = time.time() - start_time
        
        metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        
        ablation_results.append({
            'category': 'Batch Size',
            'config': f'BS={bs}',
            'accuracy': metrics['exact_accuracy'],
            'hamming': metrics['hamming_score'],
            'f1': metrics['f1'],
            'time': train_time
        })
        
        print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, "
              f"Hamming: {metrics['hamming_score']*100:.1f}%, Time: {train_time:.1f}s")
    
    plot_ablation_results(ablation_results, config.PLOTS_DIR)
    
   
    
    print(f"\n{'Category':<15} {'Config':<15} {'Accuracy':<12} {'Hamming':<12} {'F1':<10}")
    
    for r in ablation_results:
        print(f"{r['category']:<15} {r['config']:<15} "
              f"{r['accuracy']*100:>6.1f}%     {r['hamming']*100:>6.1f}%     {r['f1']:.4f}")
    
    return ablation_results


def run_cerinta4(config, dataset, test_loader, worst_fold_idx):
    
    
    stratify = np.array([l[5] for l in dataset.labels])
    kfold = StratifiedKFold(config.K_FOLDS, shuffle=True, random_state=config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(dataset)), stratify)):
        if fold == worst_fold_idx:
            break
    
    train_transform = get_base_transform(config.IMAGE_SIZE)
    val_transform = get_val_transform(config.IMAGE_SIZE)
    
    train_sub = SubsetWithTransform(dataset, train_idx, train_transform)
    val_sub = SubsetWithTransform(dataset, val_idx, val_transform)
    
    train_loader = DataLoader(train_sub, config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, config.BATCH_SIZE, num_workers=2)
    
    results_c4 = []
    criterion = nn.BCEWithLogitsLoss()
    
    
    
    model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
    
    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
    train_time = time.time() - start_time
    
    metrics = evaluate(model, val_loader, criterion, config.DEVICE)
    results_c4.append({
        'config': 'No ES, No LRS',
        'accuracy': metrics['exact_accuracy'],
        'hamming': metrics['hamming_score'],
        'f1': metrics['f1'],
        'time': train_time,
        'epochs': config.NUM_EPOCHS
    })
    print(f"  Acc: {metrics['exact_accuracy']*100:.1f}%, Time: {train_time:.1f}s")
    
    
    
    model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=3)
    
    start_time = time.time()
    epochs_ran = 0
    for epoch in range(15):
        train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        epochs_ran = epoch + 1
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    train_time = time.time() - start_time
    
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
    metrics = evaluate(model, val_loader, criterion, config.DEVICE)
    results_c4.append({
        'config': 'With Early Stopping',
        'accuracy': metrics['exact_accuracy'],
        'hamming': metrics['hamming_score'],
        'f1': metrics['f1'],
        'time': train_time,
        'epochs': epochs_ran
    })
    print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, Epochs: {epochs_ran}, Time: {train_time:.1f}s")
    
    
    
    model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_metrics['loss'])
    train_time = time.time() - start_time
    
    metrics = evaluate(model, val_loader, criterion, config.DEVICE)
    results_c4.append({
        'config': 'With LR Scheduler',
        'accuracy': metrics['exact_accuracy'],
        'hamming': metrics['hamming_score'],
        'f1': metrics['f1'],
        'time': train_time,
        'epochs': config.NUM_EPOCHS
    })
    print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, Time: {train_time:.1f}s")
    
    
    
    model = EfficientNetClassifier(dropout=0.5).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=3)
    
    start_time = time.time()
    epochs_ran = 0
    for epoch in range(15):
        train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_metrics['loss'])
        epochs_ran = epoch + 1
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    train_time = time.time() - start_time
    
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
    metrics = evaluate(model, val_loader, criterion, config.DEVICE)
    results_c4.append({
        'config': 'ES + LR Scheduler',
        'accuracy': metrics['exact_accuracy'],
        'hamming': metrics['hamming_score'],
        'f1': metrics['f1'],
        'time': train_time,
        'epochs': epochs_ran
    })
    print(f"Acc: {metrics['exact_accuracy']*100:.1f}%, Epochs: {epochs_ran}, Time: {train_time:.1f}s")
    
    
    print(f"{'Config':<25} {'Accuracy':<12} {'Hamming':<12} {'F1':<10} {'Time':<10} {'Epochs'}")
    
    for r in results_c4:
        print(f"{r['config']:<25} {r['accuracy']*100:>6.1f}%     {r['hamming']*100:>6.1f}%     "
              f"{r['f1']:.4f}    {r['time']:>6.1f}s    {r['epochs']}")
    
    return results_c4


def main():
    
    
    dataset = HemorrhageDataset(config.CSV_TRAIN, config.FOLDER_TRAIN)
    test_dataset = HemorrhageDataset(config.CSV_TEST, config.FOLDER_TEST,
                                     get_val_transform(config.IMAGE_SIZE))
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE, num_workers=2)
    
    all_experiments = {}
    
    
    
    baseline_results, worst_fold, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=False, use_oversampling=False, monai_augment=None,
        description="Baseline"
    )
    all_experiments['Baseline'] = baseline_results
    
    
    
    weights_results, _, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=True, use_oversampling=False, monai_augment=None,
        description="With Weights"
    )
    all_experiments['With_Weights'] = weights_results
    
    oversample_results, _, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=False, use_oversampling=True, monai_augment=None,
        description="With Oversampling"
    )
    all_experiments['With_Oversampling'] = oversample_results
    
    
    
    aug1_results, _, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=True, monai_augment=get_monai_augment_set1(),
        description="MONAI Light"
    )
    all_experiments['MONAI_Light'] = aug1_results
    
    aug2_results, _, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=True, monai_augment=get_monai_augment_set2(),
        description="MONAI Medium"
    )
    all_experiments['MONAI_Medium'] = aug2_results
    
    aug3_results, worst_fold_final, _, _ = run_kfold_training(
        config, dataset, test_loader,
        use_weights=True, monai_augment=get_monai_augment_set3(),
        description="MONAI Heavy"
    )
    all_experiments['MONAI_Heavy'] = aug3_results
    
    
    
    plot_experiments_comparison(all_experiments, config.PLOTS_DIR)
    create_results_table(all_experiments, config.PLOTS_DIR)
    
    
    
    cerinta4_results = run_cerinta4(config, dataset, test_loader, worst_fold_final)
    all_experiments['Cerinta4_ES_LRS'] = cerinta4_results
    
    ablation_results = run_ablation_study(config, dataset, test_loader, worst_fold_final)
    all_experiments['ablation'] = ablation_results
    
    
    
    print(f"\n{'Experiment':<20} {'Test Acc':<15} {'Test Hamming':<15} {'Test F1':<10}")
    
    
    for name, results in all_experiments.items():
        if name != 'ablation' and 'test' in results:
            acc = results['test']['exact_accuracy']['mean']
            hamm = results['test']['hamming_score']['mean']
            f1 = results['test']['f1']['mean']
            print(f"{name:<20} {acc*100:>6.1f}%        {hamm*100:>6.1f}%        {f1:.4f}")
    
    best_result = max(
        [(k, v) for k, v in all_experiments.items() if k != 'ablation' and 'test' in v],
        key=lambda x: x[1]['test']['exact_accuracy']['mean']
    )
    
    best_acc = best_result[1]['test']['exact_accuracy']['mean']
    best_hamm = best_result[1]['test']['hamming_score']['mean']
    
    
    print(f"Cel mai bun: {best_result[0]}")
    print(f"Accuracy >= 40%: {best_acc*100:.1f}% {'PASS' if best_acc >= 0.40 else 'FAIL'}")
    print(f"Hamming >= 70%:  {best_hamm*100:.1f}% {'PASS' if best_hamm >= 0.70 else 'FAIL'}")
    
    with open(f'{config.OUTPUT_DIR}/all_results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        json.dump(convert(all_experiments), f, indent=2)
    
    
    
    for f in sorted(os.listdir(config.PLOTS_DIR)):
        if f.endswith('.png') or f.endswith('.csv'):
            print(f"  {f}")
    
    
    
    return all_experiments


if __name__ == "__main__":
    
    start_time = time.time()
    
    results = main()
    
    elapsed = time.time() - start_time
    print(f"\nTimp total: {elapsed/60:.1f} minute ({elapsed/3600:.2f} ore)")
