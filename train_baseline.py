import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ESC50Dataset(Dataset):
    def __init__(self, root_dir, folds, sample_rate=44100, n_mels=64):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        csv_path = os.path.join(root_dir, "meta", "esc50.csv")
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.root_dir, "audio", row["filename"])

        waveform_np, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        mel = self.melspec(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        label = int(row["target"])
        return mel, label

class SmallCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in tqdm(loader, desc="Training", leave=False):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in tqdm(loader, desc="Evaluating", leave=False):
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root_dir = "ESC-50"

    full_train_dataset = ESC50Dataset(root_dir=root_dir, folds=[1, 2, 3, 4])
    test_dataset = ESC50Dataset(root_dir=root_dir, folds=[5])

    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_baseline.pt")
            print("Saved new best model.")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load("best_baseline.pt", map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Clean Test Loss: {test_loss:.4f}")
    print(f"Final Clean Test Acc:  {test_acc:.4f}")

if __name__ == "__main__":
    main()