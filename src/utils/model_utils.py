import torch, copy
from torch import nn
import torch.nn.functional as F
from timm.optim import create_optimizer

def create_optimizer_scheduler(model_copy, encoders, decoders, args):
    params = []
    for module in encoders.values():
        params += list(module.parameters())

    for module in decoders.values():
        params += list(module.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Set up optimizer and scheduler for fine-tuning
    optimizer_ft = create_optimizer(args, model_copy)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.epochs, eta_min=0)
    if args.warmup:
        warmup_epochs = 20
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer_ft,
        start_factor=0.1,  # Start at 10% of base lr
        end_factor=1.0,
        total_iters=warmup_epochs
    )

        # Create cosine scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=args.epochs - warmup_epochs,  # Adjust for warmup period
        eta_min=1e-6
    )

        # Combine schedulers
        scheduler_ft = torch.optim.lr_scheduler.SequentialLR(
        optimizer_ft,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    else:
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.epochs, eta_min=0)

    return optimizer, scheduler, optimizer_ft, scheduler_ft

def get_layer_names(args):
    selected_layers = []
    # Select layers for compression sequentially for number of blocks
    for i in range(args.total_blocks):
        selected_layers.extend([f'block_{i}_attn_qkv', f'block_{i}_attn_proj', 
                                f'block_{i}_mlp_fc1', f'block_{i}_mlp_fc2'])
    return selected_layers


# Need this class for forward pass with decoded weights    
class CustomFC(nn.Module):
    def __init__(self, bias):
        super(CustomFC, self).__init__()
        self.bias = nn.Parameter(bias, requires_grad=True)
        self.weight = None

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x

def clone_weights(layer):
    return layer.weight.data.clone().detach(), layer.bias.data.clone().detach()

def create_custom_fc(bias):   
    return CustomFC(bias)

def process_model_blocks(model, model_copy, args, selected_layers=None, skip_qkv=False):
    original_weights = {'qkv': [], 'proj': [], 'fc1': [], 'fc2': []}

    if skip_qkv:
        original_weights.pop('qkv')
    
    for i in range(args.total_blocks):
        attn_layer = model.blocks[i].attn
        mlp_layer = model.blocks[i].mlp
        
    
        if f'block_{i}_attn_qkv' in selected_layers:
            weight_qkv, bias_qkv = clone_weights(attn_layer.qkv)
            original_weights['qkv'].append(weight_qkv.T)

            custom_qkv = create_custom_fc(bias_qkv)
            model_copy.blocks[i].attn.qkv = custom_qkv
            
            
        if f'block_{i}_attn_proj' in selected_layers:
            weight_proj, bias_proj = clone_weights(attn_layer.proj)
            original_weights['proj'].append(weight_proj.T)
            
            custom_proj = create_custom_fc(bias_proj)
            model_copy.blocks[i].attn.proj = custom_proj
 

           
        if f'block_{i}_mlp_fc1' in selected_layers:
            weight_fc1, bias_fc1 = clone_weights(mlp_layer.fc1)
            original_weights['fc1'].append(weight_fc1.T)
            
            custom_fc1 = create_custom_fc(bias_fc1)
            model_copy.blocks[i].mlp.fc1 = custom_fc1


        if f'block_{i}_mlp_fc2' in selected_layers:
            weight_fc2, bias_fc2 = clone_weights(mlp_layer.fc2)
            original_weights['fc2'].append(weight_fc2.T)
            
            custom_fc2 = create_custom_fc(bias_fc2)
            model_copy.blocks[i].mlp.fc2 = custom_fc2
    input_seq_lens = {}
    for key in original_weights:
        if len(original_weights[key]) > 0:
            original_weights[key] = torch.cat(original_weights[key], dim=0)
            input_seq_lens[key] = original_weights[key].shape[0]
        else:
            # del original_weights[key] 
            original_weights[key] = None
            # input_seq_lens[key] = 0
    
    original_weights = {k: v for k, v in original_weights.items() if v is not None}

    return original_weights, input_seq_lens

def update_model_weights(model_copy, decoded_weights, total_blocks=12, selected_layers=None, 
                         skip_qkv=False, hidden_dim=384):
    
    if not skip_qkv and 'qkv' in decoded_weights:
        qkv_matrices = torch.split(decoded_weights['qkv'], hidden_dim, dim=0)

    if 'proj' in decoded_weights:
        proj_matrices = torch.split(decoded_weights['proj'], hidden_dim, dim=0)
    
    if 'fc1' in decoded_weights:
        fc1_matrices = torch.split(decoded_weights['fc1'], hidden_dim, dim=0)
    
    if 'fc2' in decoded_weights:
        fc2_matrices = torch.split(decoded_weights['fc2'], hidden_dim * 4, dim=0)
    
    qkv_idx = proj_idx = fc1_idx = fc2_idx = 0
    for i in range(total_blocks):
        # Update qkv weight
        if f'block_{i}_attn_qkv' in selected_layers:
            model_copy.blocks[i].attn.qkv.weight = qkv_matrices[qkv_idx].T
            qkv_idx += 1
        
        
        # Update proj weight
        if f'block_{i}_attn_proj' in selected_layers:
            model_copy.blocks[i].attn.proj.weight = proj_matrices[proj_idx].T
            proj_idx += 1
            
        
        # Update fc1 weight
        if f'block_{i}_mlp_fc1' in selected_layers:
            model_copy.blocks[i].mlp.fc1.weight = fc1_matrices[fc1_idx].T
            fc1_idx += 1
        
        
        # Update fc2 weight
        if f'block_{i}_mlp_fc2' in selected_layers:
            model_copy.blocks[i].mlp.fc2.weight = fc2_matrices[fc2_idx].T
            fc2_idx += 1

class CustomFcFlops(nn.Module):
    def __init__(self, hidden_dim, rank, out_dim, bias):
        super(CustomFcFlops, self).__init__()
        self.bias = nn.Parameter(bias, requires_grad=True)
        self.z = torch.rand(hidden_dim, rank).to(bias.device)
        self.w = torch.rand(rank, out_dim).to(bias.device)

    def forward(self, x):
        # x = F.linear(x, self.weight, self.bias)
        x = F.linear(F.linear(x, self.z.T), self.w.T, self.bias)
        return x


def create_model_for_flops(model, args, selected_layers=None):
    model_copy = copy.deepcopy(model)

    
    for i in range(args.total_blocks):
        attn_layer = model.blocks[i].attn
        mlp_layer = model.blocks[i].mlp
        
    
        if f'block_{i}_attn_qkv' in selected_layers:
            _, bias_qkv = clone_weights(attn_layer.qkv)
            custom_qkv = CustomFcFlops(hidden_dim=args.hidden_dim, rank=args.rank, 
                                       out_dim=args.hidden_dim*3, bias=bias_qkv)
            model_copy.blocks[i].attn.qkv = custom_qkv
            
            
        if f'block_{i}_attn_proj' in selected_layers:
            _, bias_proj = clone_weights(attn_layer.proj)
            custom_proj = CustomFcFlops(hidden_dim=args.hidden_dim, rank=args.rank, 
                                        out_dim=args.hidden_dim, bias=bias_proj)
            model_copy.blocks[i].attn.proj = custom_proj

           
        if f'block_{i}_mlp_fc1' in selected_layers:
            _, bias_fc1 = clone_weights(mlp_layer.fc1)
            custom_fc1 = CustomFcFlops(hidden_dim=args.hidden_dim, rank=args.rank, 
                                       out_dim=args.hidden_dim*4, bias=bias_fc1)
            model_copy.blocks[i].mlp.fc1 = custom_fc1


        if f'block_{i}_mlp_fc2' in selected_layers:
            _, bias_fc2 = clone_weights(mlp_layer.fc2)
            custom_fc2 = CustomFcFlops(hidden_dim=args.hidden_dim*4, rank=args.rank, 
                                       out_dim=args.hidden_dim, bias=bias_fc2)
            model_copy.blocks[i].mlp.fc2 = custom_fc2

    return model_copy         