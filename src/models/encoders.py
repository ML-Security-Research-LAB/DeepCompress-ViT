import torch
from torch import nn
from .quantization import ActQ

class AffineTransformer(nn.Module):
    def __init__(self, input_seq_len, input_dim):
        super(AffineTransformer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_seq_len, input_dim))
        self.W.data.fill_(1.0)

        self.B = nn.Parameter(torch.randn(input_seq_len, input_dim))
        self.B.data.fill_(0.0)
        
    def forward(self, x):
        x = x * self.W + self.B
        return x    

    def __repr__(self):
        return f'AffineTransformer{self.W.shape[0], self.W.shape[1]}'

# class Encoder(nn.Module):
#     def __init__(self, hidden_dim, rank, layer_type='qkv', modify_input=True, input_seq_len=None):
#         super(Encoder, self).__init__()        

#         assert modify_input, "Modify input must be True for this model."
#         assert input_seq_len is not None, "Input length must be provided."
#         assert layer_type in ['qkv', 'proj', 'fc1', 'fc2'], "Layer type must be one of ['qkv', 'proj', 'fc1', 'fc2']"
        
#         if layer_type == 'qkv':
#             layers = [AffineTransformer(input_seq_len, hidden_dim * 3),
#                       nn.Linear(hidden_dim * 3, hidden_dim * 3, bias=False),
#                       nn.Linear(hidden_dim * 3, hidden_dim * 2, bias=False),
#                       nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
#                       nn.Linear(hidden_dim, rank, bias=False),
#                       AffineTransformer(input_seq_len, rank)]
#         elif layer_type == 'proj':
#             layers = [AffineTransformer(input_seq_len, hidden_dim),
#                       nn.Linear(hidden_dim, hidden_dim, bias=False),
#                       nn.Linear(hidden_dim, rank, bias=False),
#                       AffineTransformer(input_seq_len, rank)]
#         elif layer_type == 'fc1':
#             layers = [AffineTransformer(input_seq_len, hidden_dim * 4),
#                       nn.Linear(hidden_dim * 4, hidden_dim * 4, bias=False),
#                       nn.Linear(hidden_dim * 4, hidden_dim * 3, bias=False),
#                       nn.Linear(hidden_dim * 3, hidden_dim * 2, bias=False),
#                       nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
#                       nn.Linear(hidden_dim, rank, bias=False),
#                       AffineTransformer(input_seq_len, rank)]
#         elif layer_type == 'fc2':
#             layers = [AffineTransformer(input_seq_len, hidden_dim),
#                       nn.Linear(hidden_dim, hidden_dim, bias=False),
#                       nn.Linear(hidden_dim, rank, bias=False),
#                       AffineTransformer(input_seq_len, rank)]
        
#         self.encoder = nn.Sequential(*layers)
#         self.act = ActQ(rank, nbits_a=4)
    
#     def forward(self, x):
#         return self.act(self.encoder(x))

class Encoder(nn.Module):
    def __init__(self, hidden_dim, rank, layer_type='qkv', modify_input=True, input_seq_len=None):
        super(Encoder, self).__init__()        

        assert modify_input, "Modify input must be True for this model."
        assert input_seq_len is not None, "Input length must be provided, It is batch of weights."
        assert layer_type in ['qkv', 'proj', 'fc1', 'fc2'], "Layer type must be one of ['qkv', 'proj', 'fc1', 'fc2']"
        
        if layer_type == 'qkv':
            layers = [AffineTransformer(input_seq_len, hidden_dim * 3),
                      nn.Linear(hidden_dim * 3, hidden_dim * 2, bias=False),
                    #   nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                      nn.Linear(hidden_dim * 2, rank, bias=False),
                      AffineTransformer(input_seq_len, rank)
                      ]
        elif layer_type == 'proj':
            layers = [AffineTransformer(input_seq_len, hidden_dim),
                      nn.Linear(hidden_dim, hidden_dim, bias=False),
                      nn.Linear(hidden_dim, rank, bias=False),
                      AffineTransformer(input_seq_len, rank)]
        elif layer_type == 'fc1':
            layers = [AffineTransformer(input_seq_len, hidden_dim * 4),
                    #   nn.Linear(hidden_dim * 4, hidden_dim * 4, bias=False),
                    #   nn.Linear(hidden_dim * 4, hidden_dim * 3, bias=False),
                      nn.Linear(hidden_dim * 4, hidden_dim * 2, bias=False),
                    #   nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                      nn.Linear(hidden_dim * 2, rank, bias=False),
                      AffineTransformer(input_seq_len, rank)]
        elif layer_type == 'fc2':
            layers = [AffineTransformer(input_seq_len, hidden_dim),
                      nn.Linear(hidden_dim, hidden_dim, bias=False),
                      nn.Linear(hidden_dim, rank, bias=False),
                      AffineTransformer(input_seq_len, rank)]
        
        self.encoder = nn.Sequential(*layers)
        self.act = ActQ(rank, nbits_a=4)
    
    def forward(self, x):
        return self.act(self.encoder(x))