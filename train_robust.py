import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


def add_noise_snr(waveform, snr_db):
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-12)
    out = waveform + noise
    out = torch.clamp(out, -1.0, 1.0)
    return out


def apply_lowpass(waveform, sr, cutoff_hz):
    out = AF.lowpass_biquad(waveform, sample_rate=sr, cutoff_freq=cutoff_hz)
    out = torch.clamp(out, -1.0, 1.0)
    return out


def apply_mulaw(waveform, quantization_channels):
    waveform = torch.clamp(waveform, -1.0, 1.0)
    encoded = AF.mu_law_encoding(waveform, quantization_channels=quantization_channels)
    decoded = AF.mu_law_decoding(encoded, quantization_channels=quantization_channels)
    decoded = torch.clamp(decoded, -1.0, 1.0)
    return decoded


def random_waveform_augment(waveform, sr):
    """
    Randomly apply ONE augmentation to the waveform.
    """
    aug_type = random.choice(["noise", "lowpass", "mulaw"])

    if aug_type == "noise":
        snr_db = random.choice([20, 10, 0])
        return add_noise_snr(waveform, snr_db)

    elif aug_type == "lowpass":
        cutoff_hz = random.choice([8000, 4000, 2000])
        return apply_lowpass(waveform, sr, cutoff_hz)

    elif aug_type == "mulaw":
        q = random.choice([256, 64, 16])
        return apply_mulaw(waveform, quantization_channels=q)

    return waveform


class ESC50Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        folds,
        sample_rate=44100,
        n_mels=64,
        train_mode=False,
        use_waveform_aug=False,
        use_specaug=False
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.train_mode = train_mode
        self.use_waveform_aug = use_waveform_aug
        self.use_specaug = use_specaug

        csv_path = os.path.join(root_dir, "meta", "esc50.csv")
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels
        )

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)

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
            sr = self.sample_rate

        # Apply waveform augmentation ONLY during training
        if self.train_mode and self.use_waveform_aug:
            if random.random() < 0.5:
                waveform = random_waveform_augment(waveform, sr)

        mel = self.melspec(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Apply SpecAugment ONLY during training
        if self.train_mode and self.use_specaug:
            if random.random() < 0.8:
                mel = self.time_mask(mel)
                mel = self.freq_mask(mel)

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
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root_dir = "ESC-50"


    use_waveform_aug = True
    use_specaug = True
    model_name = "both"

    # For specaug run:
    # use_waveform_aug = False
    # use_specaug = True
    # model_name = "specaug"

    # For both run:
    # use_waveform_aug = True
    # use_specaug = True
    # model_name = "both"

    full_train_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[1, 2, 3, 4],
        train_mode=False
    )

    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices_dataset, val_dataset_raw = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_indices = train_indices_dataset.indices
    val_indices = val_dataset_raw.indices

    all_df = full_train_dataset.df

    train_df = all_df.iloc[train_indices].reset_index(drop=True)
    val_df = all_df.iloc[val_indices].reset_index(drop=True)

    train_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[1, 2, 3, 4],
        train_mode=True,
        use_waveform_aug=use_waveform_aug,
        use_specaug=use_specaug
    )
    train_dataset.df = train_df

    val_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[1, 2, 3, 4],
        train_mode=False,
        use_waveform_aug=False,
        use_specaug=False
    )
    val_dataset.df = val_df

    test_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[5],
        train_mode=False,
        use_waveform_aug=False,
        use_specaug=False
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Config:", model_name)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    best_val_acc = 0.0
    save_path = f"best_{model_name}.pt"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("\nLoading best model for final clean test evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Clean Test Loss: {test_loss:.4f}")
    print(f"Final Clean Test Acc:  {test_acc:.4f}")


if __name__ == "__main__":
    main()