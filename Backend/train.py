import pandas as pd
import torch
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

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

class SkinDataset(Dataset):
    def __init__(self, sources, transform=None):
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

        self.df = pd.DataFrame(records)
        unique_labels = sorted(self.df['label'].unique().tolist())
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(unique_labels)}

        with open('label_mapping.json', 'w') as f:
            json.dump(self.id_to_label, f)

        self.df['target'] = self.df['label'].map(self.label_to_id)
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

transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

dataset = SkinDataset(SOURCES, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(dataset.label_to_id)
model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
model = model.to(device)

# 1. Lowered Learning Rate for Fine-Tuning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. Added Learning Rate Scheduler (Halves the LR every 4 epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

class_counts = dataset.df['target'].value_counts().sort_index().values
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum() * num_classes
weights = weights.to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=weights)

checkpoint_path = 'pause_checkpoint.pth'
best_model_path = 'skin_model_best.pth'
start_epoch = 0
start_batch = 0
best_loss = float('inf')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch_idx'] + 1
    best_loss = checkpoint.get('best_loss', float('inf'))

model.train()

try:
    for epoch in range(start_epoch, 15):
        epoch_loss = 0.0
        processed_batches = 0
        
        for batch_idx, (images, labels) in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_batches += 1
            
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} processed. Loss: {loss.item():.4f}")
            
        start_batch = 0
        scheduler.step() # Decay learning rate at the end of the epoch
        
        # 3. Save the best model
        avg_epoch_loss = epoch_loss / processed_batches if processed_batches > 0 else float('inf')
        print(f"--- Epoch {epoch} Completed | Avg Loss: {avg_epoch_loss:.4f} ---")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"New best loss achieved! Saving {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

except KeyboardInterrupt:
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'best_loss': best_loss
    }, checkpoint_path)
    exit(0)

# Save the final epoch state just in case, but rely on skin_model_best.pth for production
torch.save(model.state_dict(), 'skin_model_latest.pth')
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)