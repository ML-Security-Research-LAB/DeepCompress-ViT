import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.profiler import record_function

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def evaluate_new_mixed(val_loader, model, iters=10, mp=False):
    accs = AverageMeter()

    # switch to evaluate mode
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(val_loader, total=len(val_loader))
    for i, (input, target) in enumerate(pbar):
        target = target.cuda()
        input = input.cuda()

        with torch.cuda.amp.autocast(enabled=mp):
            output = model(input)
            
        if isinstance(output, dict):
            output = output['head']
        # output, pert_output = tor
        prediction = output.argmax(1)
        preds.append(prediction.cpu())
        targets.append(target.cpu())
       
        acc = 100 * prediction.eq(target).float().sum()/len(target)
        accs.update(acc.item(), input.size(0))
        pbar.set_postfix(acc=accs.avg)

        if i == iters:
            break
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = 100 * preds.eq(targets).float().sum()/len(targets)

    return acc.item()

def encoder_decoder_quant_params(encoders, decoders):
    alphas, zeros = 0, 0
    for key in encoders.keys():
        alphas += encoders[key].act.alpha.numel()
        zeros += encoders[key].act.zero_point.numel()
        alphas += decoders[key].alpha.numel()
    return alphas, zeros

def model_quant_params(model):
    return model.patch_embed.proj.alpha.numel() + model.head.alpha.numel()

def compare_model_parameters(model, compressed_model, encoder, decoder, 
                           original_weights, args):
    """
    Calculate and print the number of parameters before and after compression.
    
    Args:
        model: The original, uncompressed model.
        compressed_model: The compressed model.
        encoder: The encoder model.
        decoder: The decoder model.
        original_weights: The original model weights.
    """
    before_compression = 0
    after_compression = 0

    before_compression += original_weights['qkv'].numel()
    encoded = encoder['qkv'](original_weights['qkv'].to(args.device))
    after_compression += decoder['qkv'].weight.numel() + encoded.numel()
    
    before_compression += original_weights['proj'].numel()
    encoded = encoder['proj'](original_weights['proj'].to(args.device))
    after_compression += decoder['proj'].weight.numel() + encoded.numel()
   
    before_compression += original_weights['fc1'].numel()
    encoded = encoder['fc1'](original_weights['fc1'].to(args.device))
    after_compression += decoder['fc1'].weight.numel() + encoded.numel()
    
    before_compression += original_weights['fc2'].numel()
    encoded = encoder['fc2'](original_weights['fc2'].to(args.device))
    after_compression += decoder['fc2'].weight.numel() + encoded.numel()

    enc_dec_mem = (after_compression * 0.5) / 1024 ** 2

    first_last = model.patch_embed.proj.weight.data.numel() + model.head.weight.data.numel()
    compressed_mem_first_last = (first_last * 1) / 1024 ** 2

    ln_params = 0
    # bias_params = 0
    for name,param in model.named_parameters():  # Use compressed_model instead of model
        if 'norm' in name and 'weight' in name:
            ln_params += param.numel()
        elif 'cls_token' in name:
            ln_params += param.numel()
        elif '.bias' in name:
            # bias_params += param.numel()
            ln_params += param.numel()
        elif 'pos_embed' in name:
            ln_params += param.numel()
    # print('model bias params:', bias_params)

    
    ln_storage = (ln_params * 2) / 1024 ** 2
    
    alpha, zero = encoder_decoder_quant_params(encoder, decoder)
    alpha += model_quant_params(compressed_model)
    alpha_storage = (alpha * 2) / 1024 ** 2
    zero_storage = (zero * 2) / 1024 ** 2
    
    total_storage = enc_dec_mem + compressed_mem_first_last + ln_storage + alpha_storage + zero_storage
    print(f"total storage: {total_storage:.2f} MB")
    
    # Calculate total parameters
    total_parameters = sum(p.numel() for p in model.parameters())
    new_parameters = total_parameters - before_compression + after_compression

    # Print results
    print(f'Before compression: {before_compression/1e6:.2f} M')
    print(f'After compression: {after_compression/1e6:.2f} M')
    print(f"Old parameters: {total_parameters/1e6:.2f} M")
    print(f"New parameters: {new_parameters/1e6:.2f} M")
    
    return total_storage  # Return the calculated storage

def flops_comparison(layers, args):
    """
    Compare the FLOPs between the original model and the decomposed model.
    
    Args:
        layers (list): List of layer names to be compressed.
        rank (int): The rank used for low-rank decomposition.
        input_size, input_dim, qkv_dim, proj_dim, fc1_dim, fc2_dim: Model-specific dimensions.
    
    Returns:
        tuple: Original FLOPs, decomposed FLOPs, and reduction percentage.
    """
    rank = args.rank

    if 'small' in args.model_name:
        args.hidden_dim = 384
        input_size=197
        input_dim=384
        qkv_dim=1152
        proj_dim=384
        fc1_dim=1536 
        fc2_dim=1536
    elif 'base' in args.model_name:
        args.hidden_dim = 768
        input_size=197
        input_dim=768
        qkv_dim=2304
        proj_dim=768
        fc1_dim=3072 
        fc2_dim=3072

    # Calculate original FLOPs
    original_flops = sum(
        2 * input_size * (
            qkv_dim * input_dim * ('attn_qkv' in layer) +
            proj_dim * input_dim * ('attn_proj' in layer) +
            fc1_dim * input_dim * ('mlp_fc1' in layer) +
            input_dim * fc2_dim * ('mlp_fc2' in layer)
        ) for layer in layers
    )

    # Calculate decomposed FLOPs
    decomposed_flops = sum(
        2 * input_size * (
            (qkv_dim * rank + rank * input_dim) * ('attn_qkv' in layer) +
            (proj_dim * rank + rank * input_dim) * ('attn_proj' in layer) +
            (fc1_dim * rank + rank * input_dim) * ('mlp_fc1' in layer) +
            (fc2_dim * rank + rank * input_dim) * ('mlp_fc2' in layer)
        ) for layer in layers
    )

    # Calculate reduction percentage
    reduction_percentage = (original_flops - decomposed_flops) / original_flops * 100

    # Print detailed results
    print(f"Original FLOPs: {original_flops:,}")
    print(f"Decomposed FLOPs: {decomposed_flops:,}")
    print(f"Reduction: {reduction_percentage:.2f}%")

    
    return original_flops, decomposed_flops, reduction_percentage



def count_flops(model, input):
    """
    Count the number of FLOPs for a given model and input.
    
    Args:
        model: The model to count FLOPs for.
        input: The input tensor to the model.
    
    Returns:
        int: The number of FLOPs.
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=True,   # Track memory usage (optional)
        with_stack=True,       # Collect stack information
        record_shapes=True     # Record input shapes (useful for FLOPs calculation)
    ) as prof:
        with record_function("model_inference"):
            # Run the model inference
            output = model(input)

    # Sum the FLOPs from all recorded events
    total_flops = sum([event.flops for event in prof.key_averages()])
    print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")

def analyze_parameter_storage(compressed_model, encoder, decoder, original_weights, args):
    """
    Analyze and print the storage requirements of different parameter groups.
    
    Args:
        compressed_model: The compressed model.
        encoder: The encoder model.
        decoder: The decoder model.
        original_weights: The original model weights.
        args: Arguments containing device information.
    """
    storage_details = {
        '4-bit': {'components': {}, 'total': 0},
        '8-bit': {'components': {}, 'total': 0},
        '16-bit': {'components': {}, 'total': 0},
        'others': {'components': {}, 'total': 0}
    }
    
    # Calculate storage for encoded layers (4-bit)
    for key in ['qkv', 'proj', 'fc1', 'fc2']:
        encoded = encoder[key](original_weights[key].to(args.device))
        decoder_params = decoder[key].weight.numel()
        encoded_params = encoded.numel()
        
        storage_details['4-bit']['components'][f"{key}_encoded"] = (encoded_params * 0.5) / (1024 ** 2)
        storage_details['4-bit']['components'][f"{key}_decoder"] = (decoder_params * 0.5) / (1024 ** 2)
        storage_details['4-bit']['total'] += (encoded_params * 0.5 + decoder_params * 0.5) / (1024 ** 2)
    
    # First and last layer storage (8-bit)
    patch_embed_params = compressed_model.patch_embed.proj.weight.data.numel()
    head_params = compressed_model.head.weight.data.numel() 
    
    storage_details['8-bit']['components']["patch_embed"] = (patch_embed_params * 1) / (1024 ** 2)
    storage_details['8-bit']['total'] += (patch_embed_params * 1) / (1024 ** 2)
    
    if head_params > 0:
        storage_details['8-bit']['components']["head"] = (head_params * 1) / (1024 ** 2)
        storage_details['8-bit']['total'] += (head_params * 1) / (1024 ** 2)
    
    # LayerNorm and other parameters (16-bit)
    layer_norm_params = 0
    cls_token_params = 0
    bias_params = 0
    pos_embed_params = 0
    
    for name, param in compressed_model.named_parameters():
        if 'norm' in name and 'weight' in name:
            layer_norm_params += param.numel()
        elif 'cls_token' in name:
            cls_token_params += param.numel()
        elif '.bias' in name:
            bias_params += param.numel()
        elif 'pos_embed' in name:
            pos_embed_params += param.numel()
    
    if layer_norm_params > 0:
        storage_details['16-bit']['components']["layer_norm"] = (layer_norm_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (layer_norm_params * 2) / (1024 ** 2)
    
    if cls_token_params > 0:
        storage_details['16-bit']['components']["cls_token"] = (cls_token_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (cls_token_params * 2) / (1024 ** 2)
    
    if bias_params > 0:
        storage_details['16-bit']['components']["bias_params"] = (bias_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (bias_params * 2) / (1024 ** 2)
    
    if pos_embed_params > 0:
        storage_details['16-bit']['components']["pos_embed"] = (pos_embed_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (pos_embed_params * 2) / (1024 ** 2)
    
    # Quantization parameters (16-bit)
    alpha, zero = encoder_decoder_quant_params(encoder, decoder)
    alpha += model_quant_params(compressed_model)
    
    storage_details['16-bit']['components']["alpha_params"] = (alpha * 2) / (1024 ** 2)
    storage_details['16-bit']['total'] += (alpha * 2) / (1024 ** 2)
    
    storage_details['16-bit']['components']["zero_point_params"] = (zero * 2) / (1024 ** 2)
    storage_details['16-bit']['total'] += (zero * 2) / (1024 ** 2)
    
    # Calculate total storage
    total_storage = (storage_details['4-bit']['total'] + 
                     storage_details['8-bit']['total'] + 
                     storage_details['16-bit']['total'])
    
    # Print detailed report
    print("\n" + "="*70)
    print(f"{'PARAMETER STORAGE ANALYSIS':^70}")
    print("="*70)
    
    print(f"\nTOTAL STORAGE: {total_storage:.2f} MB\n")
    
    print(f"{'Bit Precision':<15}{'Storage (MB)':<15}{'Percentage':<15}{'Components'}")
    print("-"*70)
    
    # Print by bit precision
    for bit_precision in ['4-bit', '8-bit', '16-bit']:
        precision_storage = storage_details[bit_precision]['total']
        percentage = (precision_storage / total_storage) * 100
        component_count = len(storage_details[bit_precision]['components'])
        
        print(f"{bit_precision:<15}{precision_storage:<15.2f}{percentage:<15.2f}%{component_count} components")
    
    # Print all components by bit precision group
    print("\n" + "-"*70)
    print(f"{'DETAILED STORAGE BY BIT PRECISION':^70}")
    print("-"*70)
    
    # Print 4-bit components
    print(f"\n{'4-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['4-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Print 8-bit components
    print(f"\n{'8-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['8-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Print 16-bit components
    print(f"\n{'16-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['16-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Continue with existing summary
    print("\n" + "-"*70)
    print(f"{'BREAKDOWN BY COMPONENT TYPE':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Bit Precision':<15}{'Storage (MB)':<15}{'Percentage'}")
    print("-"*70)
    
    # Sort all components by storage size (descending)
    all_components = []
    for precision, details in storage_details.items():
        for component, storage in details['components'].items():
            all_components.append((component, precision, storage, (storage/total_storage)*100))
    
    all_components.sort(key=lambda x: x[2], reverse=True)
    
    # Print all components
    for component, precision, storage, percentage in all_components:
        print(f"{component:<25}{precision:<15}{storage:<15.2f}{percentage:.2f}%")
    
    # After printing all the individual components, add a final summary section:
    print("\n" + "-"*70)
    print(f"{'COMPRESSED MODEL STORAGE SUMMARY BY BIT PRECISION':^70}")
    print("-"*70)
    
    # Calculate the totals for each precision
    total_4bit = storage_details['4-bit']['total']
    total_8bit = storage_details['8-bit']['total']
    total_16bit = storage_details['16-bit']['total']
    
    # Print each precision's total storage with percentage
    print(f"{'4-bit storage:':<20}{total_4bit:.2f} MB ({(total_4bit/total_storage)*100:.2f}%)")
    print(f"{'8-bit storage:':<20}{total_8bit:.2f} MB ({(total_8bit/total_storage)*100:.2f}%)")
    print(f"{'16-bit storage:':<20}{total_16bit:.2f} MB ({(total_16bit/total_storage)*100:.2f}%)")
    print(f"{'Total storage:':<20}{total_storage:.2f} MB (100.00%)")
    
    # Calculate what the storage would be if everything was in 32-bit
    total_params = 0
    
    for key in ['qkv', 'proj', 'fc1', 'fc2']:
        encoded = encoder[key](original_weights[key].to(args.device))
        decoded = decoder[key](encoded)
        total_params += decoded.numel()
    
    patch_embed_params = compressed_model.patch_embed.proj.weight.data.numel()
    head_params = compressed_model.head.weight.data.numel()
    total_params += patch_embed_params + head_params
    
    for name, param in compressed_model.named_parameters():
        if ('norm' in name and 'weight' in name) or 'cls_token' in name or '.bias' in name or 'pos_embed' in name:
            total_params += param.numel()
        
    # Calculate 32-bit storage and compression ratio
    storage_32bit = (total_params * 4) / (1024 ** 2)  # 4 bytes per parameter
    compression_ratio = storage_32bit / total_storage
    
    print("\n" + "-"*70)
    print(f"{'COMPRESSION METRICS':^70}")
    print("-"*70)
    print(f"{'Original model size:':<20} {storage_32bit:.2f} MB (in 32-bit)")
    print(f"{'Compressed model size:':<20} {total_storage:.2f} MB")
    print(f"{'Compression ratio:':<20}{compression_ratio:.1f}x")
    
    print("\n" + "="*70)
    

def analyze_parameter_storage_mask_rcnn_vit_backbone(compressed_model, encoder, decoder, 
                                                     original_weights, args):
    """
    Analyze and print the storage requirements of different parameter groups.
    
    Args:
        compressed_model: The compressed model.
        encoder: The encoder model.
        decoder: The decoder model.
        original_weights: The original model weights.
        args: Arguments containing device information.
    """
    storage_details = {
        '4-bit': {'components': {}, 'total': 0},
        '8-bit': {'components': {}, 'total': 0},
        '16-bit': {'components': {}, 'total': 0},
        'others': {'components': {}, 'total': 0}
    }
    
    # Calculate storage for encoded layers (4-bit)
    for key in ['qkv', 'proj', 'fc1', 'fc2']:
        encoded = encoder[key](original_weights[key].to(args.device))
        decoder_params = decoder[key].weight.numel()
        encoded_params = encoded.numel()
        
        storage_details['4-bit']['components'][f"{key}_encoded"] = (encoded_params * 0.5) / (1024 ** 2)
        storage_details['4-bit']['components'][f"{key}_decoder"] = (decoder_params * 0.5) / (1024 ** 2)
        storage_details['4-bit']['total'] += (encoded_params * 0.5 + decoder_params * 0.5) / (1024 ** 2)
    
    # First and last layer storage (8-bit)
    patch_embed_params = compressed_model.patch_embed.proj.weight.data.numel()
    head_params = 0 #compressed_model.head.weight.data.numel() 
    
    storage_details['8-bit']['components']["patch_embed"] = (patch_embed_params * 1) / (1024 ** 2)
    storage_details['8-bit']['total'] += (patch_embed_params * 1) / (1024 ** 2)
    
    if head_params > 0:
        storage_details['8-bit']['components']["head"] = (head_params * 1) / (1024 ** 2)
        storage_details['8-bit']['total'] += (head_params * 1) / (1024 ** 2)
    
    # LayerNorm and other parameters (16-bit)
    layer_norm_params = 0
    cls_token_params = 0
    bias_params = 0
    pos_embed_params = 0
    
    for name, param in compressed_model.named_parameters():
        if 'norm' in name and 'weight' in name:
            layer_norm_params += param.numel()
        elif 'cls_token' in name:
            cls_token_params += param.numel()
        elif '.bias' in name and 'head' not in name:  # Exclude head bias if it exists
            bias_params += param.numel()
        elif 'pos_embed' in name:
            pos_embed_params += param.numel()
    
    additional_layer_norm_params = 0#(768.0 * 4) # Adding the layer norm params for the backbone
    layer_norm_params += additional_layer_norm_params

    if layer_norm_params > 0:
        storage_details['16-bit']['components']["layer_norm"] = (layer_norm_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (layer_norm_params * 2) / (1024 ** 2)
    
    if cls_token_params > 0:
        storage_details['16-bit']['components']["cls_token"] = (cls_token_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (cls_token_params * 2) / (1024 ** 2)
    
    if bias_params > 0:
        storage_details['16-bit']['components']["bias_params"] = (bias_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (bias_params * 2) / (1024 ** 2)
    
    if pos_embed_params > 0:
        storage_details['16-bit']['components']["pos_embed"] = (pos_embed_params * 2) / (1024 ** 2)
        storage_details['16-bit']['total'] += (pos_embed_params * 2) / (1024 ** 2)
    
    # Quantization parameters (16-bit)
    alpha, zero = encoder_decoder_quant_params(encoder, decoder)
    alpha += compressed_model.patch_embed.proj.alpha.numel()#model_quant_params(compressed_model)
    
    storage_details['16-bit']['components']["alpha_params"] = (alpha * 2) / (1024 ** 2)
    storage_details['16-bit']['total'] += (alpha * 2) / (1024 ** 2)
    
    storage_details['16-bit']['components']["zero_point_params"] = (zero * 2) / (1024 ** 2)
    storage_details['16-bit']['total'] += (zero * 2) / (1024 ** 2)
    
    # add params of up projection layers
    up1_proj_params = 0#1181184.0 # number of params in the up projection layers
    up2_proj_params = 0#590208.0 # number of params in the up projection layers
    # up1
    storage_details['16-bit']['total'] += (up1_proj_params * 2) / (1024 ** 2) # 1181184 is the number of params in the up1 projection layers
    # up2
    storage_details['16-bit']['total'] += (up2_proj_params * 2) / (1024 ** 2) # 590208 is the number of params in the up2 projection layers
    
    # Calculate total storage
    total_storage = (storage_details['4-bit']['total'] + 
                     storage_details['8-bit']['total'] + 
                     storage_details['16-bit']['total'])
    
    # Print detailed report
    print("\n" + "="*70)
    print(f"{'PARAMETER STORAGE ANALYSIS':^70}")
    print("="*70)
    
    print(f"\nTOTAL STORAGE: {total_storage:.2f} MB\n")
    
    print(f"{'Bit Precision':<15}{'Storage (MB)':<15}{'Percentage':<15}{'Components'}")
    print("-"*70)
    
    # Print by bit precision
    for bit_precision in ['4-bit', '8-bit', '16-bit']:
        precision_storage = storage_details[bit_precision]['total']
        percentage = (precision_storage / total_storage) * 100
        component_count = len(storage_details[bit_precision]['components'])
        
        print(f"{bit_precision:<15}{precision_storage:<15.2f}{percentage:<15.2f}%{component_count} components")
    
    # Print all components by bit precision group
    print("\n" + "-"*70)
    print(f"{'DETAILED STORAGE BY BIT PRECISION':^70}")
    print("-"*70)
    
    # Print 4-bit components
    print(f"\n{'4-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['4-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Print 8-bit components
    print(f"\n{'8-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['8-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Print 16-bit components
    print(f"\n{'16-BIT COMPONENTS':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Storage (MB)':<15}{'Percentage of Total'}")
    print("-"*70)
    
    for component, storage in sorted(storage_details['16-bit']['components'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (storage / total_storage) * 100
        print(f"{component:<25}{storage:<15.2f}{percentage:.2f}%")
    
    # Continue with existing summary
    print("\n" + "-"*70)
    print(f"{'BREAKDOWN BY COMPONENT TYPE':^70}")
    print("-"*70)
    print(f"{'Component':<25}{'Bit Precision':<15}{'Storage (MB)':<15}{'Percentage'}")
    print("-"*70)
    
    # Sort all components by storage size (descending)
    all_components = []
    for precision, details in storage_details.items():
        for component, storage in details['components'].items():
            all_components.append((component, precision, storage, (storage/total_storage)*100))
    
    all_components.sort(key=lambda x: x[2], reverse=True)
    
    # Print all components
    for component, precision, storage, percentage in all_components:
        print(f"{component:<25}{precision:<15}{storage:<15.2f}{percentage:.2f}%")
    
    # After printing all the individual components, add a final summary section:
    print("\n" + "-"*70)
    print(f"{'COMPRESSED MODEL STORAGE SUMMARY BY BIT PRECISION':^70}")
    print("-"*70)
    
    # Calculate the totals for each precision
    total_4bit = storage_details['4-bit']['total']
    total_8bit = storage_details['8-bit']['total']
    total_16bit = storage_details['16-bit']['total']
    
    # Print each precision's total storage with percentage
    print(f"{'4-bit storage:':<20}{total_4bit:.2f} MB ({(total_4bit/total_storage)*100:.2f}%)")
    print(f"{'8-bit storage:':<20}{total_8bit:.2f} MB ({(total_8bit/total_storage)*100:.2f}%)")
    print(f"{'16-bit storage:':<20}{total_16bit:.2f} MB ({(total_16bit/total_storage)*100:.2f}%)")
    print(f"{'Total storage:':<20}{total_storage:.2f} MB (100.00%)")
    
    # Calculate what the storage would be if everything was in 32-bit
    total_params = 0
    
    for key in ['qkv', 'proj', 'fc1', 'fc2']:
        encoded = encoder[key](original_weights[key].to(args.device))
        decoded = decoder[key](encoded)
        total_params += decoded.numel()
    
    patch_embed_params = compressed_model.patch_embed.proj.weight.data.numel()
    # head_params = compressed_model.head.weight.data.numel()
    total_params += patch_embed_params #+ head_params
    
    for name, param in compressed_model.named_parameters():
        if ('norm' in name and 'weight' in name) or 'cls_token' in name or '.bias' in name or 'pos_embed' in name:
            # print(f"param name: {name}, numel: {param.numel()}")
            if 'head' in name:
                continue
            total_params += param.numel()
        
    total_params += additional_layer_norm_params
    total_params += up1_proj_params + up2_proj_params  # Add params of up projection layers
    # Calculate 32-bit storage and compression ratio
    storage_32bit = (total_params * 4) / (1024 ** 2)  # 4 bytes per parameter
    compression_ratio = storage_32bit / total_storage
    
    print("\n" + "-"*70)
    print(f"{'COMPRESSION METRICS':^70}")
    print("-"*70)
    print(f"{'Original model size:':<20} {storage_32bit:.2f} MB (in 32-bit)")
    print(f"{'Compressed model size:':<20} {total_storage:.2f} MB")
    print(f"{'Compression ratio:':<20}{compression_ratio:.1f}x")
    
    print("\n" + "="*70)

