# HuBERT model classes.
#
# Mirrors Wav2Vec2/model.py exactly, swapping Wav2Vec2Model for HubertModel.
# PretrainedHubertModel  — wraps HuggingFace HubertModel, returns (z, c)
# Prediction             — device-aware inference wrapper (same interface)

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class PretrainedHubertModel(nn.Module):
    """HuggingFace HuBERT wrapper.

    HuBERT uses the same Wav2Vec2FeatureExtractor for input preprocessing.
    The model backbone is HubertModel; output structure mirrors Wav2Vec2Model:
        outputs.extract_features  -> CNN features  (z, [T, 512])
        outputs.last_hidden_state -> Transformer    (c, [T, 768])

    Adapted from the Wav2Vec2 equivalent (pipeline.ipynb cells 4-5)
    with HubertModel substituted for Wav2Vec2Model.
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path
        )
        self.model = HubertModel.from_pretrained(model_name_or_path)
        self.model.eval()

    def forward(self, wav: np.ndarray, sr: int):
        inputs = self.feature_extractor(
            wav, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():
            outputs = self.model(input_values)
        # Some transformers versions do not expose `extract_features` on HubertModel output.
        # Fallback to the model's convolutional feature extractor for z.
        if hasattr(outputs, "extract_features") and outputs.extract_features is not None:
            z = outputs.extract_features.squeeze(0).cpu().numpy()
        else:
            z_tensor = self.model.feature_extractor(input_values)
            z = z_tensor.transpose(1, 2).squeeze(0).detach().cpu().numpy()
        c = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return z, c


class Prediction:
    """Device-aware HuBERT inference wrapper — same interface as Wav2Vec2/model.py."""

    def __init__(self, model_name_or_path: str, gpu: int = 0):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.model  = PretrainedHubertModel(model_name_or_path)
        self.model.to(self.device)

    def __call__(self, wav: np.ndarray, sr: int):
        return self.model(wav, sr)
