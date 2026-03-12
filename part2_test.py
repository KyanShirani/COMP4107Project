import os
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, random_split

# -------------------------
# Step A: Dataset class
# -------------------------
class ESC50Dataset(Dataset):
    def __init__(self, root_dir, folds, sample_rate=44100, n_mels=64):
        """
        root_dir: path to ESC-50 folder
        folds: list like [1,2,3,4]
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Load CSV metadata
        csv_path = os.path.join(root_dir, "meta", "esc50.csv")
        self.df = pd.read_csv(csv_path)

        # Keep only requested folds
        self.df = self.df[self.df["fold"].isin(folds)].reset_index(drop=True)

        # Mel-spectrogram transform
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

        # Build full path to audio file
        audio_path = os.path.join(self.root_dir, "audio", row["filename"])

        # Load waveform with soundfile instead of torchaudio.load
        waveform_np, sr = sf.read(audio_path, dtype="float32")

        # Convert numpy array to torch tensor
        waveform = torch.tensor(waveform_np, dtype=torch.float32)

        # Make shape [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if sample rate is different
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to Mel-spectrogram
        mel = self.melspec(waveform)  # [1, n_mels, time]

        # Log scale
        mel = torch.log(mel + 1e-6)

        # Normalize per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Label
        label = int(row["target"])

        return mel, label


# -------------------------
# Step B: Main test
# -------------------------
def main():
    root_dir = "ESC-50"   # change this if needed

    # Standard split:
    # train = folds 1,2,3,4
    # test  = fold 5
    full_train_dataset = ESC50Dataset(root_dir=root_dir, folds=[1, 2, 3, 4])
    test_dataset = ESC50Dataset(root_dir=root_dir, folds=[5])

    # Make validation split from training set
    val_size = int(0.15 * len(full_train_dataset))   # 15% for validation
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Print dataset sizes
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # Check one sample
    mel, label = train_dataset[0]
    print("One sample mel shape:", mel.shape)
    print("One sample label:", label)


if __name__ == "__main__":
    main()