import os
import timm
import torch
import argparse
from tqdm import tqdm
from src.models import get_models, get_encoder_decoders
from src.utils import (seed_everything, get_dataloaders_imagenet, get_dataloaders_cifar10, compare_model_parameters, evaluate_new_mixed, 
                       get_layer_names, update_model_weights, process_model_blocks, create_model_for_flops, count_flops,
                       analyze_parameter_storage, analyze_parameter_storage_mask_rcnn_vit_backbone)


# path = '/home/sahmed9/codes/ViT-Compression-Latest/saved_models/small_all_layers_qvit_quant/deit_small_patch16_224.pth'

def change_to_half(compressed_model):
    # Iterate through all modules in the model
    for module in compressed_model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()

    # Convert pos_embed to half if it exists in the model
    if hasattr(compressed_model, 'pos_embed'):
        compressed_model.pos_embed.data = compressed_model.pos_embed.data.half()
        
    if hasattr(compressed_model, 'cls_token'):
        compressed_model.cls_token.data = compressed_model.cls_token.data.half()
    
    # Convert all bias parameters to half precision
    for name, param in compressed_model.named_parameters():
        if 'bias' in name and param is not None:
            param.data = param.data.half()

def main(args):
    seed_everything()
    torch.cuda.set_device(args.device)
    
    state = torch.load(args.state_path, map_location=args.device)

    args.hidden_dim = 768 if 'base' in args.model_name else 384

    # Load and prepare the model
    model, compressed_model = get_models(args)

    selected_layers = get_layer_names(args)

    # extract original weights and input sequence lengths (total batch of weights)
    original_weights, input_seq_lens = process_model_blocks(model, compressed_model, args, selected_layers=selected_layers, 
                                                            skip_qkv=args.skip_qkv)

    # Get data loaders
    if args.dataset == 'cifar10':
        _, val_loader = get_dataloaders_cifar10(args.batch_size)
    elif args.dataset == 'imagenet':
        _, val_loader = get_dataloaders_imagenet(args.batch_size)

    # FLOPs calculation
    print('Original model:')
    count_flops(model, torch.rand(1,3,224,224).cuda())
    model_flops = create_model_for_flops(model, args, selected_layers) 
           
    print('Compressed model:')
    count_flops(model_flops, torch.rand(1,3,224,224).cuda())
    del model_flops
    
    global encoders, decoders

    encoders, decoders = get_encoder_decoders(selected_layers, input_seq_lens, args)
    
    for key in encoders.keys():
        print(f'Encoder {key}: {encoders[key]}')
        encoders[key].load_state_dict(state['encoder_states'][key])
        decoders[key].load_state_dict(state['decoder_states'][key])
        
    compressed_model.load_state_dict(state['model_state_dict'])
    change_to_half(compressed_model)

    compare_model_parameters(model, compressed_model, encoders, 
                             decoders, original_weights, args)
    
    analyze_parameter_storage(compressed_model, encoders, decoders, original_weights, args)
    
    if args.mask_rcnn_backbone:
        print('Analyzing parameter storage for Mask R-CNN backbone...')
        analyze_parameter_storage_mask_rcnn_vit_backbone(compressed_model, encoders, decoders, original_weights, args)
    
    decoded_weights = {}
    for key in decoders.keys():
        decoded_weights[key] = decoders[key](encoders[key](original_weights[key].to(args.device)))
    
    update_model_weights(compressed_model, decoded_weights, total_blocks=args.total_blocks, selected_layers=selected_layers,
                         skip_qkv=args.skip_qkv, hidden_dim=args.hidden_dim)

    best_acc = evaluate_new_mixed(val_loader, compressed_model, iters=-1, mp=args.mixed_precision)

    print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer with weight compression')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Name of the Vision Transformer model')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'], help='Dataset to use for training')
    parser.add_argument('--total_blocks', type=int, default=12, help='Number of blocks to compress')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for ImageNet dataloader')
    parser.add_argument('--state_path', type=str, default='saved_models/small_rank_277/deit_small_patch16_224.pth', help='Path to the model state dict')
    parser.add_argument('--skip_qkv', action='store_true', help='Skip compressing qkv layer in MultiheadAttention')
    parser.add_argument('--rank', type=int, default=200, help='Rank for encoded weight matrices')
    parser.add_argument('--mask-rcnn-backbone', action='store_true', help='Show storage for Mask R-CNN backbone')
    args = parser.parse_args()
    main(args)
