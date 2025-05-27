import timm
import torch
from .quantization import Conv2dQ, LinearQ

def quantize_first_and_last_layer(model, args, nbits=8):
    """
    Replaces patch embedding projection and head layers with their quantized versions.
    
    Args:
        model: The model to be modified
        args: Arguments containing device information
        nbits: Number of bits for quantization (default: 8)
    
    Returns:
        model: The modified model with quantized layers
    """
    # Quantize patch embedding projection
    patch_embed_proj = model.patch_embed.proj
    model.patch_embed.proj = Conv2dQ(
        patch_embed_proj.in_channels,
        patch_embed_proj.out_channels,
        patch_embed_proj.kernel_size,
        stride=patch_embed_proj.stride,
        padding=patch_embed_proj.padding,
        dilation=patch_embed_proj.dilation,
        groups=patch_embed_proj.groups,
        nbits=nbits
    ).to(args.device)
    
    # Copy weights and biases for patch embedding
    model.patch_embed.proj.weight.data = patch_embed_proj.weight.data.clone()
    model.patch_embed.proj.bias.data = patch_embed_proj.bias.data.clone()
    
    # Quantize head layer
    head = model.head
    model.head = LinearQ(
        head.in_features,
        head.out_features,
        bias=True,
        nbits=nbits
    ).to(args.device)
    
    # Copy weights and biases for head
    model.head.weight.data = head.weight.data.clone()
    model.head.bias.data = head.bias.data.clone()
    
    return model

def get_models(args):
    model = timm.create_model(args.model_name, pretrained=True)
    model.cuda()

    compressed_model = timm.create_model(args.model_name, pretrained=True)
    compressed_model.cuda()

    compressed_model = quantize_first_and_last_layer(compressed_model, args)
    return model, compressed_model
