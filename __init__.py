"""
ComfyUI_Scalogram
=================
Custom nodes that generate Scalogram images from audio using the
Continuous Wavelet Transform (CWT).

Nodes
-----
  Scalogram from Audio   – AUDIO input  → IMAGE
  Scalogram from File    – file path    → IMAGE
  Load Audio File        – file path    → AUDIO
"""

from .nodes import ScalogramFromAudio, ScalogramFromFile, LoadAudioFile

NODE_CLASS_MAPPINGS = {
    "ScalogramFromAudio": ScalogramFromAudio,
    "ScalogramFromFile":  ScalogramFromFile,
    "LoadAudioFile":      LoadAudioFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScalogramFromAudio": "Scalogram from Audio",
    "ScalogramFromFile":  "Scalogram from File",
    "LoadAudioFile":      "Load Audio File",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
