# Wav2Vec2 model classes.
#
# Adapted from pipeline.ipynb cells 4-5 (fairseq reference L34-92).
#
# PretrainedWav2Vec2Model  — wraps HuggingFace Wav2Vec2Model, returns (z, c)
# Prediction               — device-aware inference wrapper (same interface as notebook)

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class PretrainedWav2Vec2Model(nn.Module):
    """HuggingFace Wav2Vec2.0 wrapper.

    Adapted from:
        external/fairseq/examples/wav2vec/wav2vec_featurize.py  L34-50

    Original class name: PretrainedWav2VecModel (wav2vec 1.0)
    Our class name     : PretrainedWav2Vec2Model (wav2vec 2.0)

    Original L38: model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task
    Our equivalent: Wav2Vec2Model.from_pretrained(model_name_or_path)

    Original L46: z = self.model.feature_extractor(x)   -> CNN output 512-d
    Our equivalent: outputs.extract_features              (z, not used by default)

    Original L48: c = self.model.feature_aggregator(z)  -> Transformer output
    Our equivalent: outputs.last_hidden_state             (c, 768-d, used downstream)
    """

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path
        )
        self.model = Wav2Vec2Model.from_pretrained(model_name_or_path)
        self.model.eval()                                          # L40: model.eval()

    def forward(self, wav: np.ndarray, sr: int):
        inputs = self.feature_extractor(
            wav, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(
            next(self.model.parameters()).device
        )
        with torch.no_grad():                                      # L45
            outputs = self.model(input_values)
        z = outputs.extract_features.squeeze(0).cpu().numpy()     # L47  CNN feats
        c = outputs.last_hidden_state.squeeze(0).cpu().numpy()    # L49  transformer
        return z, c


class Prediction:
    """Device-aware inference wrapper.

    Adapted from:
        external/fairseq/examples/wav2vec/wav2vec_featurize.py  L80-92

    L83-85 original:
        self.gpu   = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)
    Changed .cuda(gpu) -> device-agnostic (runs on CPU when no CUDA).

    L87-92 original (kept in spirit):
        def __call__(self, x):
            x = torch.from_numpy(x).float().cuda(self.gpu)
            with torch.no_grad():
                z, c = self.model(x.unsqueeze(0))
            return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()
    The tensor casting / device step is now inside PretrainedWav2Vec2Model.forward().
    """

    def __init__(self, model_name_or_path: str, gpu: int = 0):    # L83
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.model  = PretrainedWav2Vec2Model(model_name_or_path)
        self.model.to(self.device)

    def __call__(self, wav: np.ndarray, sr: int):                  # L87
        return self.model(wav, sr)
