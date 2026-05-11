# src/preprocessing/audio_preprocessor.py

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, sr=16000, n_mels=80, n_fft=512, 
                 hop_length=160, win_length=400):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def extract_from_video(self, video_path: Path, output_dir: Path):
        """Extract audio track from video using FFmpeg subprocess."""
        import subprocess
        audio_path = output_dir / (video_path.stem + ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-ar", str(self.sr), "-ac", "1",
            "-vn", str(audio_path)
        ], check=True, capture_output=True)
        return audio_path

    def compute_lfcc(self, audio_path: Path) -> np.ndarray:
        """Linear Frequency Cepstral Coefficients for AASIST input."""
        y, _ = librosa.load(str(audio_path), sr=self.sr)
        # Pad or trim to 4 seconds (64000 samples)
        if len(y) < 64000:
            y = np.pad(y, (0, 64000 - len(y)))
        else:
            y = y[:64000]
        
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, 
                                    hop_length=self.hop_length,
                                    win_length=self.win_length))
        # Linear filterbank (not mel)
        linear_filters = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            norm=None, htk=False
        )
        lfcc = np.dot(linear_filters, stft)
        lfcc = librosa.power_to_db(lfcc)
        return lfcc.astype(np.float32)

    def compute_mel_spectrogram(self, audio_path: Path) -> np.ndarray:
        """Mel spectrogram for RawNet2 input."""
        y, _ = librosa.load(str(audio_path), sr=self.sr)
        if len(y) < 64000:
            y = np.pad(y, (0, 64000 - len(y)))
        else:
            y = y[:64000]
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return librosa.power_to_db(mel).astype(np.float32)