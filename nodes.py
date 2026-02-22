"""
ComfyUI_Scalogram - Custom nodes for generating scalogram images from audio
using Continuous Wavelet Transform (CWT).

Nodes:
  - ScalogramFromAudio  : AUDIO type → IMAGE (scalogram)
  - ScalogramFromFile   : file path → IMAGE (scalogram)
  - LoadAudioFile       : file path → AUDIO type
"""

import os
import io
import numpy as np
import torch
from PIL import Image

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import matplotlib
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")          # non-interactive, thread-safe
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[ComfyUI_Scalogram] matplotlib not found → pip install matplotlib")

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("[ComfyUI_Scalogram] PyWavelets not found → pip install PyWavelets")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[ComfyUI_Scalogram] librosa not found → pip install librosa")

# ── Constants ─────────────────────────────────────────────────────────────────

WAVELET_OPTIONS = [
    "cmor1.5-1.0",   # Complex Morlet  – best time-frequency localisation
    "morl",          # Real Morlet
    "mexh",          # Mexican Hat (Ricker wavelet)
    "cgau4",         # Complex Gaussian – 4th derivative
    "cgau1",         # Complex Gaussian – 1st derivative
    "gaus1",         # Real Gaussian    – 1st derivative
]

COLORMAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma",
    "hot", "jet", "turbo", "coolwarm", "RdYlBu_r", "gnuplot2",
]

# CWT is O(N·S·log N) per scale; cap the signal length for interactive use
MAX_SAMPLES = 4096

# ── Internal helpers ──────────────────────────────────────────────────────────

def _require(*flags_and_names):
    """Raise RuntimeError if any required package is missing."""
    mapping = {
        "librosa": (LIBROSA_AVAILABLE,    "pip install librosa"),
        "pywt":    (PYWT_AVAILABLE,       "pip install PyWavelets"),
        "mpl":     (MATPLOTLIB_AVAILABLE, "pip install matplotlib"),
    }
    for key in flags_and_names:
        ok, hint = mapping[key]
        if not ok:
            pkg = key if key != "mpl" else "matplotlib"
            raise RuntimeError(f"[ComfyUI_Scalogram] {pkg} is required: {hint}")


def _audio_dict_to_numpy(audio: dict):
    """ComfyUI AUDIO dict → (y: np.ndarray[T], sr: int)."""
    waveform = audio["waveform"]        # [B, C, T]
    sr = int(audio["sample_rate"])
    y = waveform[0].float().mean(dim=0).cpu().numpy()   # mono [T]
    return y, sr


def _load_file_to_numpy(path: str):
    """Audio file → (y: np.ndarray[T], sr: int) via librosa."""
    _require("librosa")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ComfyUI_Scalogram] File not found: {path}")
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def _compute_scalogram(y: np.ndarray, sr: int, wavelet: str, num_scales: int):
    """
    Compute the CWT scalogram of a 1-D signal.

    Returns
    -------
    power_norm  : np.ndarray [num_scales, T]  normalised power in [0, 1]
                  Row 0 = highest frequency (small scale).
    frequencies : np.ndarray [num_scales]      Hz, descending order.
    n_samples   : int   number of (possibly resampled) samples
    sr_out      : int   (possibly resampled) sample rate
    """
    _require("pywt")

    # ── Downsample if too long ────────────────────────────────────────────────
    if len(y) > MAX_SAMPLES:
        if LIBROSA_AVAILABLE:
            target_sr = max(1, int(sr * MAX_SAMPLES / len(y)))
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        else:
            step = max(1, len(y) // MAX_SAMPLES)
            y = y[::step][:MAX_SAMPLES]
            sr = sr // step

    # ── Log-spaced scales: small scale → high freq, large scale → low freq ───
    scales = np.geomspace(1, num_scales, num_scales)

    # ── CWT ──────────────────────────────────────────────────────────────────
    dt = 1.0 / sr
    coefficients, frequencies = pywt.cwt(y, scales, wavelet, sampling_period=dt)
    # coefficients : [num_scales, T],  frequencies : descending (high→low)

    # ── Power → dB → normalise ───────────────────────────────────────────────
    power    = np.abs(coefficients) ** 2
    power_db = 10.0 * np.log10(power + 1e-10)
    p_min, p_max = power_db.min(), power_db.max()
    power_norm = (power_db - p_min) / max(p_max - p_min, 1e-10)

    return power_norm, frequencies, len(y), sr


def _render_to_tensor(
    power_norm: np.ndarray,
    frequencies: np.ndarray,
    n_samples: int,
    sr: int,
    colormap: str,
    image_width: int,
    image_height: int,
    log_frequency: bool,
    show_axes: bool,
) -> torch.Tensor:
    """
    Render the scalogram to a ComfyUI IMAGE tensor [1, H, W, C] ∈ [0, 1].

    power_norm rows: row 0 = highest frequency (small scale).
    For display, we flip so that low freq is at the bottom (standard convention).
    """
    _require("mpl")

    dpi   = 100
    fig_w = image_width  / dpi
    fig_h = image_height / dpi

    # Flip for display: low freq at bottom (pcolormesh y ascending)
    freqs_asc  = frequencies[::-1]          # ascending
    power_flip = power_norm[::-1, :]        # matching flip

    time_axis  = np.linspace(0.0, n_samples / sr, n_samples)

    if show_axes:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        im = ax.pcolormesh(time_axis, freqs_asc, power_flip,
                           cmap=colormap, shading="auto",
                           vmin=0.0, vmax=1.0)
        if log_frequency and freqs_asc.min() > 0:
            ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Scalogram (CWT)")
        plt.colorbar(im, ax=ax, label="Normalised Power")
        plt.tight_layout()
        bg = "white"
        bbox = "tight"
    else:
        # Borderless: axes fill the entire figure canvas
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.pcolormesh(time_axis, freqs_asc, power_flip,
                      cmap=colormap, shading="auto",
                      vmin=0.0, vmax=1.0)
        if log_frequency and freqs_asc.min() > 0:
            ax.set_yscale("log")
        ax.axis("off")
        bg   = "black"
        bbox = None

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi,
                bbox_inches=bbox, facecolor=bg)
    buf.seek(0)
    plt.close(fig)

    img = Image.open(buf).convert("RGB")
    img = img.resize((image_width, image_height), Image.LANCZOS)

    arr    = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W, C]
    return tensor


# ── Node definitions ──────────────────────────────────────────────────────────

class ScalogramFromAudio:
    """
    Generate a Scalogram image from a ComfyUI AUDIO input.

    Uses the Continuous Wavelet Transform (CWT) which provides better
    time-frequency resolution than the STFT-based spectrogram,
    especially at low frequencies.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "wavelet": (WAVELET_OPTIONS, {
                    "default": "cmor1.5-1.0",
                    "tooltip": "Wavelet family. Complex Morlet (cmor) is recommended.",
                }),
                "num_scales": ("INT", {
                    "default": 128, "min": 8, "max": 512, "step": 8,
                    "tooltip": "Number of frequency scales. More = finer detail, slower.",
                }),
                "colormap": (COLORMAP_OPTIONS, {"default": "viridis"}),
                "image_width":  ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "image_height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 64}),
                "log_frequency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Logarithmic frequency axis (recommended for music/speech).",
                }),
                "show_axes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overlay time and frequency axis labels.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("scalogram",)
    FUNCTION      = "generate"
    CATEGORY      = "audio/analysis"
    DESCRIPTION   = "Generates a CWT scalogram image from a ComfyUI AUDIO input."

    def generate(self, audio, wavelet, num_scales, colormap,
                 image_width, image_height, log_frequency, show_axes):
        _require("pywt", "mpl")
        y, sr = _audio_dict_to_numpy(audio)
        power_norm, frequencies, n_samples, sr = _compute_scalogram(
            y, sr, wavelet, num_scales)
        tensor = _render_to_tensor(
            power_norm, frequencies, n_samples, sr,
            colormap, image_width, image_height, log_frequency, show_axes)
        return (tensor,)


class ScalogramFromFile:
    """
    Load an audio file and generate its Scalogram image in one node.

    Accepts WAV, MP3, FLAC, OGG, and any other format supported by librosa.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to the audio file (WAV, MP3, FLAC, …).",
                }),
                "wavelet": (WAVELET_OPTIONS, {"default": "cmor1.5-1.0"}),
                "num_scales": ("INT", {
                    "default": 128, "min": 8, "max": 512, "step": 8,
                }),
                "colormap": (COLORMAP_OPTIONS, {"default": "viridis"}),
                "image_width":  ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "image_height": ("INT", {"default": 512,  "min": 64, "max": 4096, "step": 64}),
                "log_frequency": ("BOOLEAN", {"default": True}),
                "show_axes":     ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("scalogram",)
    FUNCTION      = "generate"
    CATEGORY      = "audio/analysis"
    DESCRIPTION   = "Loads an audio file and generates its CWT scalogram image."

    def generate(self, audio_path, wavelet, num_scales, colormap,
                 image_width, image_height, log_frequency, show_axes):
        _require("librosa", "pywt", "mpl")
        y, sr = _load_file_to_numpy(audio_path)
        power_norm, frequencies, n_samples, sr = _compute_scalogram(
            y, sr, wavelet, num_scales)
        tensor = _render_to_tensor(
            power_norm, frequencies, n_samples, sr,
            colormap, image_width, image_height, log_frequency, show_axes)
        return (tensor,)


class LoadAudioFile:
    """
    Load an audio file from disk and output it as a ComfyUI AUDIO type.

    Compatible with all standard ComfyUI audio nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to the audio file.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "load"
    CATEGORY      = "audio"
    DESCRIPTION   = "Load an audio file and output it as a ComfyUI AUDIO tensor."

    def load(self, audio_path):
        _require("librosa")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"[ComfyUI_Scalogram] File not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=None, mono=False)

        if y.ndim == 1:
            y = y[np.newaxis, :]        # [1, T]

        waveform = torch.from_numpy(y.copy()).unsqueeze(0).float()  # [1, C, T]
        return ({"waveform": waveform, "sample_rate": sr},)
