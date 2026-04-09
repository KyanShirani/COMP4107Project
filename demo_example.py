import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt



ROOT_DIR = "ESC-50"
BASELINE_MODEL_PATH = "best_baseline.pt"
ROBUST_MODEL_PATH = "best_waveform_aug.pt"   # best overall model from results

SAMPLE_RATE = 44100
N_MELS = 64

NOISE_SNR_DB = 0   # heavy noise


DEMO_FILENAME =  "5-182404-A-18.wav"



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


melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

def load_waveform(audio_path):
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
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE

    return waveform, sr


def waveform_to_logmel(waveform):
    mel = melspec_transform(waveform)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel


def add_noise_snr(waveform, snr_db):
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-12)
    noisy = waveform + noise
    noisy = torch.clamp(noisy, -1.0, 1.0)
    return noisy


@torch.no_grad()
def predict_from_waveform(model, waveform, device):
    model.eval()
    mel = waveform_to_logmel(waveform).unsqueeze(0).to(device)   # [1, 1, n_mels, time]
    logits = model(mel)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    return pred_idx, confidence, probs


def load_label_mapping(root_dir):
    csv_path = os.path.join(root_dir, "meta", "esc50.csv")
    df = pd.read_csv(csv_path)
    label_to_name = dict(zip(df["target"], df["category"]))
    return df, label_to_name


def load_model(model_path, device):
    model = SmallCNN(num_classes=50).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model



@torch.no_grad()
def find_good_demo_sample(root_dir, baseline_model, robust_model, device):
    """
    Search fold 5 for a strong demo example:
    - baseline correct on clean
    - baseline wrong on heavy noise
    - robust correct on heavy noise

    If none exists, return the best improvement candidate.
    """
    df, label_to_name = load_label_mapping(root_dir)
    test_df = df[df["fold"] == 5].reset_index(drop=True)

    strict_match = None
    best_fallback = None
    best_score = -1e9

    for _, row in test_df.iterrows():
        filename = row["filename"]
        true_label = int(row["target"])
        audio_path = os.path.join(root_dir, "audio", filename)

        waveform, _ = load_waveform(audio_path)
        noisy_waveform = add_noise_snr(waveform, NOISE_SNR_DB)

        # clean baseline
        base_clean_pred, base_clean_conf, base_clean_probs = predict_from_waveform(
            baseline_model, waveform, device
        )

        # noisy baseline
        base_noisy_pred, base_noisy_conf, base_noisy_probs = predict_from_waveform(
            baseline_model, noisy_waveform, device
        )

        # noisy robust
        robust_noisy_pred, robust_noisy_conf, robust_noisy_probs = predict_from_waveform(
            robust_model, noisy_waveform, device
        )

        base_clean_correct = (base_clean_pred == true_label)
        base_noisy_correct = (base_noisy_pred == true_label)
        robust_noisy_correct = (robust_noisy_pred == true_label)

        # strict ideal case
        if base_clean_correct and (not base_noisy_correct) and robust_noisy_correct:
            strict_match = filename
            break

        # fallback score
        true_prob_base_noisy = float(base_noisy_probs[true_label].item())
        true_prob_robust_noisy = float(robust_noisy_probs[true_label].item())

        score = 0.0
        if base_clean_correct:
            score += 2.0
        if robust_noisy_correct:
            score += 2.0
        if not base_noisy_correct:
            score += 2.0
        score += (true_prob_robust_noisy - true_prob_base_noisy)

        if score > best_score:
            best_score = score
            best_fallback = filename

    return strict_match if strict_match is not None else best_fallback


# =========================
# DISPLAY
# =========================
def print_prediction_block(title, pred_idx, confidence, label_to_name):
    print(f"{title}: {label_to_name[pred_idx]} (confidence = {confidence:.4f})")


def show_spectrograms(clean_waveform, noisy_waveform):
    clean_mel = waveform_to_logmel(clean_waveform).squeeze(0).numpy()
    noisy_mel = waveform_to_logmel(noisy_waveform).squeeze(0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(clean_mel, aspect="auto", origin="lower")
    plt.title("Clean Log-Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Bin")

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_mel, aspect="auto", origin="lower")
    plt.title(f"Heavy Noise Log-Mel Spectrogram (SNR={NOISE_SNR_DB} dB)")
    plt.xlabel("Time")
    plt.ylabel("Mel Bin")

    plt.tight_layout()
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df, label_to_name = load_label_mapping(ROOT_DIR)

    baseline_model = load_model(BASELINE_MODEL_PATH, device)
    robust_model = load_model(ROBUST_MODEL_PATH, device)

    # choose demo sample
    if DEMO_FILENAME is None:
        print("Searching for a good demo example...")
        filename = find_good_demo_sample(ROOT_DIR, baseline_model, robust_model, device)
    else:
        filename = DEMO_FILENAME

    print(f"\nDemo filename: {filename}")

    row = df[df["filename"] == filename].iloc[0]
    true_label = int(row["target"])
    true_name = label_to_name[true_label]

    audio_path = os.path.join(ROOT_DIR, "audio", filename)
    clean_waveform, _ = load_waveform(audio_path)
    noisy_waveform = add_noise_snr(clean_waveform, NOISE_SNR_DB)

    # Predictions on clean
    base_clean_pred, base_clean_conf, _ = predict_from_waveform(baseline_model, clean_waveform, device)
    robust_clean_pred, robust_clean_conf, _ = predict_from_waveform(robust_model, clean_waveform, device)

    # Predictions on noisy
    base_noisy_pred, base_noisy_conf, _ = predict_from_waveform(baseline_model, noisy_waveform, device)
    robust_noisy_pred, robust_noisy_conf, _ = predict_from_waveform(robust_model, noisy_waveform, device)

    print("\n" + "=" * 60)
    print("DEMO EXAMPLE")
    print("=" * 60)
    print(f"True label: {true_name}")
    print(f"Filename:   {filename}")

    print("\nCLEAN SAMPLE")
    print_prediction_block("Baseline", base_clean_pred, base_clean_conf, label_to_name)
    print_prediction_block("Robust  ", robust_clean_pred, robust_clean_conf, label_to_name)

    print("\nHEAVY NOISE SAMPLE")
    print_prediction_block("Baseline", base_noisy_pred, base_noisy_conf, label_to_name)
    print_prediction_block("Robust  ", robust_noisy_pred, robust_noisy_conf, label_to_name)

    print("\nShowing spectrograms...")
    show_spectrograms(clean_waveform, noisy_waveform)
    sf.write("demo_clean.wav", clean_waveform.squeeze(0).numpy(), SAMPLE_RATE)
    sf.write("demo_noisy.wav", noisy_waveform.squeeze(0).numpy(), SAMPLE_RATE)

    print("Saved clean audio to demo_clean.wav")
    print("Saved noisy audio to demo_noisy.wav")
    # Optional notebook audio playback:
    # from IPython.display import Audio, display
    # print("\nPlaying clean audio:")
    # display(Audio(clean_waveform.squeeze(0).numpy(), rate=SAMPLE_RATE))
    # print("Playing noisy audio:")
    # display(Audio(noisy_waveform.squeeze(0).numpy(), rate=SAMPLE_RATE))


if __name__ == "__main__":

    main()


