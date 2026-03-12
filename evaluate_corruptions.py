import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Corruption functions
# -------------------------
def add_noise_snr(waveform, snr_db):
    """
    waveform: [1, samples]
    Adds Gaussian noise at a chosen SNR.
    """
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-12)
    out = waveform + noise
    out = torch.clamp(out, -1.0, 1.0)
    return out


def apply_reverb(waveform, sr, decay_seconds=0.08, wet_mix=0.3):
    """
    Simple synthetic reverb using an exponentially decaying random impulse response.
    waveform: [1, samples]
    """
    rir_length = int(sr * decay_seconds)
    if rir_length < 2:
        return waveform

    t = torch.arange(rir_length, dtype=waveform.dtype) / sr
    decay = torch.exp(-t / decay_seconds)

    rir = torch.randn(rir_length, dtype=waveform.dtype) * decay
    rir[0] = 1.0  # keep direct path

    # normalize impulse response
    rir = rir / (rir.abs().sum() + 1e-8)

    # conv1d does correlation, so flip kernel for convolution
    kernel = rir.flip(0).view(1, 1, -1)

    x = waveform.unsqueeze(0)  # [1, 1, samples]
    wet = F.conv1d(x, kernel, padding=rir_length - 1).squeeze(0)

    # trim back to original length
    wet = wet[:, :waveform.size(1)]

    out = (1.0 - wet_mix) * waveform + wet_mix * wet
    out = out / (out.abs().max() + 1e-8)  # normalize
    return out


def apply_lowpass(waveform, sr, cutoff_hz):
    """
    Simulate bandwidth loss.
    """
    out = AF.lowpass_biquad(waveform, sample_rate=sr, cutoff_freq=cutoff_hz)
    out = torch.clamp(out, -1.0, 1.0)
    return out


def apply_mulaw(waveform, quantization_channels):
    """
    Simulate telephony / quantization distortion.
    waveform should be in [-1, 1]
    """
    waveform = torch.clamp(waveform, -1.0, 1.0)
    encoded = AF.mu_law_encoding(waveform, quantization_channels=quantization_channels)
    decoded = AF.mu_law_decoding(encoded, quantization_channels=quantization_channels)
    decoded = torch.clamp(decoded, -1.0, 1.0)
    return decoded


def get_corruption_fn(corruption_name=None, severity=None):
    """
    Returns a function f(waveform, sr) -> corrupted_waveform
    """

    if corruption_name is None:
        return None

    if corruption_name == "noise":
        snr_map = {
            "light": 20,
            "medium": 10,
            "heavy": 0,
        }
        snr_db = snr_map[severity]
        return lambda waveform, sr: add_noise_snr(waveform, snr_db)

    if corruption_name == "reverb":
        params = {
            "light":  {"decay_seconds": 0.04, "wet_mix": 0.20},
            "medium": {"decay_seconds": 0.08, "wet_mix": 0.35},
            "heavy":  {"decay_seconds": 0.14, "wet_mix": 0.50},
        }
        p = params[severity]
        return lambda waveform, sr: apply_reverb(
            waveform, sr,
            decay_seconds=p["decay_seconds"],
            wet_mix=p["wet_mix"]
        )

    if corruption_name == "lowpass":
        cutoff_map = {
            "light": 8000,
            "medium": 4000,
            "heavy": 2000,
        }
        cutoff = cutoff_map[severity]
        return lambda waveform, sr: apply_lowpass(waveform, sr, cutoff)

    if corruption_name == "mulaw":
        q_map = {
            "light": 256,
            "medium": 64,
            "heavy": 16,
        }
        q = q_map[severity]
        return lambda waveform, sr: apply_mulaw(waveform, quantization_channels=q)

    raise ValueError(f"Unknown corruption_name: {corruption_name}")


# -------------------------
# Dataset
# -------------------------
class ESC50Dataset(Dataset):
    def __init__(self, root_dir, folds, sample_rate=44100, n_mels=64, corruption_fn=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.corruption_fn = corruption_fn

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

        # load with soundfile
        waveform_np, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        # shape -> [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        # mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate

        # apply corruption on waveform BEFORE mel-spectrogram
        if self.corruption_fn is not None:
            waveform = self.corruption_fn(waveform, sr)

        # mel pipeline
        mel = self.melspec(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        label = int(row["target"])
        return mel, label


# -------------------------
# Same baseline model
# -------------------------
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


# -------------------------
# Evaluation
# -------------------------
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


def run_one_condition(model, root_dir, corruption_name, severity, device):
    corruption_fn = get_corruption_fn(corruption_name, severity)

    dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[5],  # same test fold as before
        corruption_fn=corruption_fn
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    loss, acc = evaluate(model, loader, device)
    return loss, acc


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root_dir = "ESC-50"

    # load saved baseline model
    model = SmallCNN(num_classes=50).to(device)
    model.load_state_dict(torch.load("best_baseline.pt", map_location=device))
    print("Loaded best_baseline.pt")

    results = []

    # clean
    clean_dataset = ESC50Dataset(root_dir=root_dir, folds=[5], corruption_fn=None)
    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=False, num_workers=0)
    clean_loss, clean_acc = evaluate(model, clean_loader, device)
    results.append(("clean", "-", clean_loss, clean_acc))

    # corruption conditions
    severities = ["light", "medium", "heavy"]
    corruption_types = ["noise", "lowpass", "mulaw"]

    for corruption_name in corruption_types:
        for severity in severities:
            print(f"Running: {corruption_name} | {severity}")
            loss, acc = run_one_condition(model, root_dir, corruption_name, severity, device)
            results.append((corruption_name, severity, loss, acc))

    # print table
    print("\n" + "=" * 60)
    print("Corruption Evaluation Results")
    print("=" * 60)
    print(f"{'Condition':<12} {'Severity':<10} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 60)

    for cond, sev, loss, acc in results:
        print(f"{cond:<12} {sev:<10} {loss:<12.4f} {acc:<12.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()