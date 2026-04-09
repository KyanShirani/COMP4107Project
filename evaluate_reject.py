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

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        if self.corruption_fn is not None:
            waveform = self.corruption_fn(waveform, self.sample_rate)
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

@torch.no_grad()
def evaluate_with_reject(model, loader, device, tau):
    model.eval()

    total = 0
    accepted = 0
    rejected = 0
    correct_accepted = 0
    unsafe_wrong = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        probs = F.softmax(logits, dim=1)

        max_probs, preds = probs.max(dim=1)
        accept_mask = max_probs >= tau
        reject_mask = ~accept_mask

        total += y.size(0)
        accepted += accept_mask.sum().item()
        rejected += reject_mask.sum().item()
        correct_accepted += ((preds == y) & accept_mask).sum().item()
        unsafe_wrong += ((preds != y) & accept_mask).sum().item()

    accepted_accuracy = correct_accepted / max(accepted, 1)
    rejection_rate = rejected / max(total, 1)
    unsafe_error_rate = unsafe_wrong / max(total, 1)
    coverage = accepted / max(total, 1)

    return {
        "tau": tau,
        "accepted_accuracy": accepted_accuracy,
        "rejection_rate": rejection_rate,
        "unsafe_error_rate": unsafe_error_rate,
        "coverage": coverage,
        "accepted": accepted,
        "rejected": rejected,
        "total": total,
    }


def build_validation_dataset(root_dir, corruption_fn=None):
    """
    Rebuilds the SAME validation split used before:
    folds 1-4, then 15% validation with seed 42.
    """
    full_train_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[1, 2, 3, 4],
        corruption_fn=corruption_fn
    )

    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    _, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    val_indices = val_subset.indices
    val_df = full_train_dataset.df.iloc[val_indices].reset_index(drop=True)

    val_dataset = ESC50Dataset(
        root_dir=root_dir,
        folds=[1, 2, 3, 4],
        corruption_fn=corruption_fn
    )
    val_dataset.df = val_df
    return val_dataset


def tune_threshold_practical(model, clean_val_loader, noisy_val_loader, device, clean_reject_cap=0.10):
    """
    Choose tau so that clean validation rejection rate is capped,
    then minimize unsafe error on heavy-noise validation.

    clean_reject_cap=0.10 means clean reject rate must be <= 10%.
    """
    candidates = [round(0.00 + 0.05 * i, 2) for i in range(17)]  # 0.00 ... 0.80
    rows = []

    for tau in candidates:
        clean_stats = evaluate_with_reject(model, clean_val_loader, device, tau)
        noisy_stats = evaluate_with_reject(model, noisy_val_loader, device, tau)

        row = {
            "tau": tau,
            "clean_reject_rate": clean_stats["rejection_rate"],
            "clean_accepted_acc": clean_stats["accepted_accuracy"],
            "noise_reject_rate": noisy_stats["rejection_rate"],
            "noise_accepted_acc": noisy_stats["accepted_accuracy"],
            "noise_unsafe_error": noisy_stats["unsafe_error_rate"],
        }
        rows.append(row)

    valid_rows = [r for r in rows if r["clean_reject_rate"] <= clean_reject_cap]

    if len(valid_rows) > 0:
        best = min(
            valid_rows,
            key=lambda r: (
                r["noise_unsafe_error"],       # minimize unsafe errors on heavy noise
                -r["noise_accepted_acc"],      # prefer better accepted accuracy
                r["clean_reject_rate"]         # prefer rejecting fewer clean samples
            )
        )
    else:
        # fallback: choose threshold with lowest clean rejection, then lowest noise unsafe error
        best = min(
            rows,
            key=lambda r: (
                r["clean_reject_rate"],
                r["noise_unsafe_error"],
                -r["noise_accepted_acc"]
            )
        )

    return best, rows

def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root_dir = "ESC-50"
    model_path = "best_baseline.pt"

    model = SmallCNN(num_classes=50).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded {model_path}")


    clean_val_dataset = build_validation_dataset(root_dir, corruption_fn=None)
    clean_val_loader = DataLoader(clean_val_dataset, batch_size=64, shuffle=False, num_workers=0)


    heavy_noise_fn = get_corruption_fn("noise", "heavy")
    noisy_val_dataset = build_validation_dataset(root_dir, corruption_fn=heavy_noise_fn)
    noisy_val_loader = DataLoader(noisy_val_dataset, batch_size=64, shuffle=False, num_workers=0)


    best_tau_stats, tuning_rows = tune_threshold_practical(
        model,
        clean_val_loader,
        noisy_val_loader,
        device,
        clean_reject_cap=0.10
    )

    print("\nThreshold tuning with CLEAN rejection cap <= 10%")
    print(
        f"{'tau':<8} {'clean_reject':<15} {'clean_acc':<15} "
        f"{'noise_reject':<15} {'noise_acc':<15} {'noise_unsafe':<15}"
    )
    print("-" * 90)
    for row in tuning_rows:
        print(
            f"{row['tau']:<8.2f} "
            f"{row['clean_reject_rate']:<15.4f} "
            f"{row['clean_accepted_acc']:<15.4f} "
            f"{row['noise_reject_rate']:<15.4f} "
            f"{row['noise_accepted_acc']:<15.4f} "
            f"{row['noise_unsafe_error']:<15.4f}"
        )

    chosen_tau = best_tau_stats["tau"]
    print(f"\nChosen tau: {chosen_tau:.2f}")

    test_conditions = [
        ("clean", "-", None),
        ("noise", "heavy", get_corruption_fn("noise", "heavy")),
        ("lowpass", "heavy", get_corruption_fn("lowpass", "heavy")),
        ("mulaw", "heavy", get_corruption_fn("mulaw", "heavy")),
    ]

    print("\n" + "=" * 100)
    print("Reject Option Results")
    print("=" * 100)
    print(
        f"{'Condition':<24} {'Tau':<8} {'Accepted Acc':<15} "
        f"{'Reject Rate':<15} {'Unsafe Error':<15}"
    )
    print("-" * 100)

    for name, severity, corruption_fn in test_conditions:
        test_dataset = ESC50Dataset(
            root_dir=root_dir,
            folds=[5],
            corruption_fn=corruption_fn
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        no_reject = evaluate_with_reject(model, test_loader, device, tau=0.0)
        with_reject = evaluate_with_reject(model, test_loader, device, tau=chosen_tau)

        label = name if severity == "-" else f"{name}-{severity}"

        print(
            f"{label + ' (tau=0)':<24} "
            f"{0.0:<8.2f} "
            f"{no_reject['accepted_accuracy']:<15.4f} "
            f"{no_reject['rejection_rate']:<15.4f} "
            f"{no_reject['unsafe_error_rate']:<15.4f}"
        )

        print(
            f"{label + f' (tau={chosen_tau:.2f})':<24} "
            f"{chosen_tau:<8.2f} "
            f"{with_reject['accepted_accuracy']:<15.4f} "
            f"{with_reject['rejection_rate']:<15.4f} "
            f"{with_reject['unsafe_error_rate']:<15.4f}"
        )

        print("-" * 100)

if __name__ == "__main__":
    main()