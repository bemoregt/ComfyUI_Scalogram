# ComfyUI_Scalogram

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that generate **scalogram** images from audio using the **Continuous Wavelet Transform (CWT)**.

A scalogram is a time-frequency representation similar to a spectrogram, but built on wavelets rather than the Short-Time Fourier Transform (STFT). This gives it adaptive time-frequency resolution — high temporal resolution at high frequencies, high frequency resolution at low frequencies — making it particularly well-suited for music, speech, and transient analysis.


![ 예시](https://github.com/bemoregt/ComfyUI_Scalogram/blob/main/ScrShot%202.png)

![ 예시](https://github.com/bemoregt/ComfyUI_Scalogram/blob/main/ScrShot%203.png)

![ 예시](https://github.com/bemoregt/ComfyUI_Scalogram/blob/main/ScrShot%204.png)

---

## Nodes

### Scalogram from Audio (`audio/analysis`)

Takes a ComfyUI **AUDIO** type and outputs a scalogram **IMAGE**.

| Input | Type | Default | Description |
|---|---|---|---|
| `audio` | `AUDIO` | — | Audio input from any ComfyUI audio node |
| `wavelet` | dropdown | `cmor1.5-1.0` | Wavelet family (see table below) |
| `num_scales` | INT | `128` | Number of frequency scales (8–512) |
| `colormap` | dropdown | `viridis` | Matplotlib colormap |
| `image_width` | INT | `1024` | Output image width in pixels |
| `image_height` | INT | `512` | Output image height in pixels |
| `log_frequency` | BOOLEAN | `True` | Logarithmic frequency axis |
| `show_axes` | BOOLEAN | `False` | Overlay time/frequency axis labels |

**Output:** `IMAGE` — scalogram tensor `[1, H, W, C]` in `[0, 1]`

---

### Scalogram from File (`audio/analysis`)

Loads an audio file from disk and outputs a scalogram **IMAGE** in a single node.

| Input | Type | Default | Description |
|---|---|---|---|
| `audio_path` | STRING | `""` | Full path to audio file (WAV, MP3, FLAC, OGG, …) |
| `wavelet` | dropdown | `cmor1.5-1.0` | Wavelet family |
| `num_scales` | INT | `128` | Number of frequency scales |
| `colormap` | dropdown | `viridis` | Matplotlib colormap |
| `image_width` | INT | `1024` | Output image width in pixels |
| `image_height` | INT | `512` | Output image height in pixels |
| `log_frequency` | BOOLEAN | `True` | Logarithmic frequency axis |
| `show_axes` | BOOLEAN | `False` | Overlay time/frequency axis labels |

**Output:** `IMAGE` — scalogram tensor `[1, H, W, C]` in `[0, 1]`

---

### Load Audio File (`audio`)

Loads an audio file from disk and outputs it as a ComfyUI **AUDIO** type, compatible with all standard ComfyUI audio nodes.

| Input | Type | Default | Description |
|---|---|---|---|
| `audio_path` | STRING | `""` | Full path to audio file |

**Output:** `AUDIO` — `{"waveform": Tensor[1, C, T], "sample_rate": int}`

---

## Wavelet Reference

| Name | Type | Best for |
|---|---|---|
| `cmor1.5-1.0` | Complex Morlet | General time-frequency analysis, music, speech **(recommended)** |
| `morl` | Real Morlet | General purpose, simpler alternative to cmor |
| `mexh` | Mexican Hat | Transient detection, sharp attacks |
| `cgau4` | Complex Gaussian (4th deriv.) | Smooth broadband signals |
| `cgau1` | Complex Gaussian (1st deriv.) | Edge-sensitive analysis |
| `gaus1` | Real Gaussian (1st deriv.) | Lightweight, monotonic trends |

---

## Installation

### 1. Place the folder in ComfyUI's custom nodes directory

```bash
cp -r ComfyUI_Scalogram /path/to/ComfyUI/custom_nodes/
```

Or clone directly:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI_Scalogram.git
```

### 2. Install Python dependencies

```bash
pip install -r /path/to/ComfyUI/custom_nodes/ComfyUI_Scalogram/requirements.txt
```

Or install individually:

```bash
pip install numpy torch Pillow matplotlib PyWavelets librosa
```

### 3. Restart ComfyUI

The nodes will appear under the **`audio/analysis`** and **`audio`** categories in the node browser.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `torch` | ComfyUI tensor format |
| `Pillow` | Image conversion and resizing |
| `matplotlib` | Scalogram rendering |
| `PyWavelets` | Continuous Wavelet Transform |
| `librosa` | Audio file loading and resampling |

---

## Technical Details

**Pipeline:**

```
Audio signal
    │
    ▼
Resample to ≤ 4096 samples   (for interactive performance)
    │
    ▼
Log-spaced scales via np.geomspace(1, num_scales, num_scales)
    │
    ▼
pywt.cwt(y, scales, wavelet, sampling_period=1/sr)
    │
    ▼
Power = |coefficients|²  →  dB = 10·log₁₀(power)  →  normalise to [0, 1]
    │
    ▼
Render with matplotlib (pcolormesh, optional log y-axis)
    │
    ▼
Resize to (image_width × image_height)  →  ComfyUI IMAGE tensor [1, H, W, C]
```

**Performance note:** The CWT is computed via FFT internally (`pywt`), making it O(S·N·log N) where S is the number of scales and N is the signal length. Signals longer than 4096 samples are automatically resampled before the transform.

**Output modes:**
- `show_axes=False` — borderless image filling the full canvas (suitable for use in generative pipelines)
- `show_axes=True` — labeled axes with time (s), frequency (Hz), and a normalised power colorbar

---

## Example Workflows

**Minimal (file → scalogram):**
```
ScalogramFromFile  →  PreviewImage
```

**With other audio nodes:**
```
LoadAudioFile  →  [audio processing nodes]  →  ScalogramFromAudio  →  PreviewImage
```

**Save to disk:**
```
ScalogramFromFile  →  SaveImage
```

---

## License

MIT
