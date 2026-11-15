import os
import json
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

LABEL_MAP = {
    'akiec': 'actinic keratosis',
    'bcc': 'basal cell carcinoma',
    'bkl': 'seborrheic keratosis',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'nevocytic nevus',
    'vasc': 'vascular lesion',
    'ACK': 'actinic keratosis',
    'BCC': 'basal cell carcinoma',
    'MEL': 'melanoma',
    'NEV': 'nevocytic nevus',
    'SCC': 'squamous cell carcinoma',
    'SEK': 'seborrheic keratosis',
    'AK': 'actinic keratosis',
    'BKL': 'seborrheic keratosis',
    'DF': 'dermatofibroma',
    'NV': 'nevocytic nevus',
    'VASC': 'vascular lesion',
}

def resolve_image_path(img_dir, filename_or_id):
    direct = os.path.join(img_dir, filename_or_id)
    if os.path.exists(direct):
        return direct
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
        candidate = os.path.join(img_dir, f"{filename_or_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(img_dir, f"{filename_or_id}.jpg")

def build_dataframe(sources):
    records = []
    seen_isic_ids = set()

    fitz = sources.get('fitzpatrick17k')
    if fitz and os.path.exists(fitz['csv']):
        df_fitz = pd.read_csv(fitz['csv']).dropna(subset=['md5hash', 'label'])
        for _, row in df_fitz.iterrows():
            raw_label = str(row['label']).strip()
            records.append({
                'img_path': resolve_image_path(fitz['img_dir'], str(row['md5hash']).strip()),
                'label': LABEL_MAP.get(raw_label, raw_label)
            })

    ham = sources.get('ham10000')
    if ham and os.path.exists(ham['csv']):
        df_ham = pd.read_csv(ham['csv']).dropna(subset=['image_id', 'dx'])
        for _, row in df_ham.iterrows():
            img_id = str(row['image_id']).strip()
            raw_dx = str(row['dx']).strip()
            seen_isic_ids.add(img_id)
            records.append({
                'img_path': resolve_image_path(ham['img_dir'], img_id),
                'label': LABEL_MAP.get(raw_dx, raw_dx)
            })

    pad = sources.get('pad_ufes_20')
    if pad and os.path.exists(pad['csv']):
        df_pad = pd.read_csv(pad['csv']).dropna(subset=['img_id', 'diagnostic'])
        for _, row in df_pad.iterrows():
            img_id = str(row['img_id']).strip()
            raw_diag = str(row['diagnostic']).strip()
            records.append({
                'img_path': resolve_image_path(pad['img_dir'], img_id),
                'label': LABEL_MAP.get(raw_diag, raw_diag)
            })

    isic = sources.get('isic2019')
    if isic and os.path.exists(isic['csv']):
        df_isic = pd.read_csv(isic['csv'])
        isic_cols = [c for c in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'] if c in df_isic.columns]
        for _, row in df_isic.iterrows():
            img_id = str(row['image']).strip()
            if img_id in seen_isic_ids:
                continue
            matched = [col for col in isic_cols if row[col] == 1.0]
            if matched:
                raw_code = matched[0]
                records.append({
                    'img_path': resolve_image_path(isic['img_dir'], img_id),
                    'label': LABEL_MAP.get(raw_code, raw_code)
                })

    return pd.DataFrame(records)

class EvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['img_path']
        try:
            image = Image.open(img_name).convert('RGB')
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (288, 288))
        label = self.df.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def generate_evaluation_json(model, val_loader, id_to_label, device, output_file='evaluation_metrics.json'):
    model.eval()
    all_preds = []
    all_targets = []

    print("Running validation inference to compute F1 and Confusion Matrix...")
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    num_classes = len(id_to_label)
    labels_range = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    total_samples = int(cm.sum())

    overall_accuracy = round(float(accuracy_score(y_true, y_pred)) * 100, 2)
    macro_f1 = round(float(f1_score(y_true, y_pred, labels=labels_range, average='macro', zero_division=0)) * 100, 2)
    weighted_f1 = round(float(f1_score(y_true, y_pred, labels=labels_range, average='weighted', zero_division=0)) * 100, 2)

    per_class_f1 = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)

    classes_dict = {}
    for idx in labels_range:
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total_samples - (tp + fp + fn))
        support = int(cm[idx, :].sum())

        row_copy = cm[idx, :].copy()
        row_copy[idx] = 0
        top_confused_indices = np.argsort(row_copy)[::-1][:3]
        top_confused = [
            {"label": id_to_label[int(c_idx)], "count": int(row_copy[c_idx])}
            for c_idx in top_confused_indices if row_copy[c_idx] > 0
        ]

        classes_dict[str(idx)] = {
            "label": id_to_label[idx],
            "f1_score": round(float(per_class_f1[idx]) * 100, 2),
            "precision": round(float(per_class_prec[idx]) * 100, 2),
            "recall": round(float(per_class_rec[idx]) * 100, 2),
            "support": support,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            },
            "top_confused_with": top_confused
        }

    metrics_payload = {
        "overall": {
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "total_validation_samples": total_samples
        },
        "classes": classes_dict
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved {output_file}! Overall Accuracy: {overall_accuracy}% | Weighted F1: {weighted_f1}% | Macro F1: {macro_f1}%")

if __name__ == "__main__":
    SOURCES = {
        'fitzpatrick17k': {
            'csv': '../Dataset/fitzpatrick17k/fitzpatrick17k.csv',
            'img_dir': '../Dataset/fitzpatrick17k/images'
        },
        'ham10000': {
            'csv': '../Dataset/ham10000/HAM10000_metadata.csv',
            'img_dir': '../Dataset/ham10000/images'
        },
        'isic2019': {
            'csv': '../Dataset/isic2019/ISIC_2019_Training_GroundTruth.csv',
            'img_dir': '../Dataset/isic2019/images'
        },
        'pad_ufes_20': {
            'csv': '../Dataset/pad_ufes_20/padufes.csv',
            'img_dir': '../Dataset/pad_ufes_20/images'
        }
    }

    with open('label_mapping.json', 'r', encoding='utf-8') as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    label_to_id = {v: k for k, v in id_to_label.items()}

    df = build_dataframe(SOURCES)
    df['target'] = df['label'].map(label_to_id)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    _, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['target'])

    val_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = EvalDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=len(id_to_label))

    weights_file = 'skin_model_best.pth' if os.path.exists('skin_model_best.pth') else 'skin_model.pth'
    print(f"Loading weights from {weights_file} on {device}...")
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model = model.to(device)

    generate_evaluation_json(model, val_loader, id_to_label, device)