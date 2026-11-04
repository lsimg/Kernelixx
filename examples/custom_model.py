"""Run: python -m kernelixx optimize --model examples.custom_model:load_model \
--input-shape 8,16 --method quantization --device cpu
"""
from torch import nn


def load_model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
