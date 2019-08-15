"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.redr_encoder import ReDREncoder


str2enc = {"redr": ReDREncoder}

__all__ = ["EncoderBase", "RNNEncoder", "ReDREncoder", "str2enc"]
