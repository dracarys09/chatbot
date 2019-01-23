import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference

        # Define layers


    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        pass
