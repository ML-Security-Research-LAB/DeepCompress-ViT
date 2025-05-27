import torch
import torch.nn as nn
from .encoders import Encoder
from .quantization import LinearQ

def get_encoder_decoders(selected_layers, input_seq_lens, args):
    hidden_dim, rank = args.hidden_dim, args.rank
    encoders = {}
    decoders = {}
    for layer in selected_layers:
        if 'qkv' in layer and 'qkv' not in encoders:
            encoders['qkv'] = Encoder(hidden_dim, rank, 'qkv', True, input_seq_lens['qkv'])
            decoders['qkv'] = LinearQ(rank, hidden_dim * 3, bias=False, nbits_w=4)
        if 'proj' in layer and 'proj' not in encoders:
            encoders['proj'] = Encoder(hidden_dim, rank, 'proj', True, input_seq_lens['proj'])
            decoders['proj'] = LinearQ(rank, hidden_dim, bias=False, nbits_w=4)
        if 'fc1' in layer and 'fc1' not in encoders:
            encoders['fc1'] = Encoder(hidden_dim, rank, 'fc1', True, input_seq_lens['fc1'])
            decoders['fc1'] = LinearQ(rank, hidden_dim * 4, bias=False, nbits_w=4)
        if 'fc2' in layer and 'fc2' not in encoders:
            encoders['fc2'] = Encoder(hidden_dim, rank, 'fc2', True, input_seq_lens['fc2'])
            decoders['fc2'] = LinearQ(rank, hidden_dim, bias=False, nbits_w=4)

    for key in encoders.keys():
        encoders[key].to(args.device)
        decoders[key].to(args.device)

    return encoders, decoders


