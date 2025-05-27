import torch
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from src.utils import update_model_weights
from src.utils import AverageMeter, evaluate_new_mixed  
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup



# Setup mixup function arguments
m_args = Namespace(
    mixup=0.8,
    cutmix=1.0,
    cutmix_minmax=None,
    mixup_prob=1.0,
    mixup_switch_prob=0.5,
    mixup_mode='batch',
    smoothing=0.1,
    nb_classes=1000  # ImageNet with 1000 classes
)

mixup_fn = Mixup(
    mixup_alpha=m_args.mixup,
    cutmix_alpha=m_args.cutmix,
    cutmix_minmax=m_args.cutmix_minmax,
    prob=m_args.mixup_prob,
    switch_prob=m_args.mixup_switch_prob,
    mode=m_args.mixup_mode,
    label_smoothing=m_args.smoothing,
    num_classes=m_args.nb_classes)



def initial_fit(encoders, decoders, original_weights, args):
    """
    Perform initial fitting of the decoder to approximate the original weights.
    
    Args:
        decoder: List of decoder weight tensors.
        original_weights: Original model weights.
        model_copy: Copy of the model with decomposed weights.
        selected_layers: List of layers selected for compression.
        args: Command-line arguments containing model configuration.
    """
    params = []
    for module in encoders.values():
        params += list(module.parameters())

    for module in decoders.values():
        params += list(module.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr * 10)
    # optimizer = torch.optim.Adam(params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.initial_iters, eta_min=1e-6)

    for key in original_weights.keys():
        encoders[key].train()
        decoders[key].train()

    print('Initial fitting...')
    pbar = tqdm(range(args.initial_iters))
    for _ in pbar:
        optimizer.zero_grad()
        loss = 0

        for key,value in original_weights.items():
            encoded_value = encoders[key](value.to(args.device))
            decoded_value = decoders[key](encoded_value)
            loss += F.mse_loss(decoded_value, value)

            pbar.set_postfix({'Loss': loss.item()})

        loss.backward()
        optimizer.step()
        scheduler.step()


def calculate_distillation_loss(original_model, predictions, images, T=1.0):
    with torch.no_grad():
        target_predictions = original_model(images)
        
    distillation_loss = F.kl_div(
                F.log_softmax(predictions / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                # F.log_softmax(target_predictions['head'] / T, dim=1),
                F.log_softmax(target_predictions / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / predictions.numel()
    return distillation_loss
            
def knowledge_distillation_loss(teacher_model, inputs, outputs_kd, T=1.0):
    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)
    distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
    return distillation_loss

def train_one_epoch(epoch, encoders, decoders, original_model, model_copy, original_weights, train_loader, val_loader,
                    optimizer, scheduler, scaler, args, csv_metrics, best_acc, optimizer_finetune=None, scheduler_finetune=None, selected_layers=None):
    pbar = tqdm(train_loader, total=len(train_loader))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model_copy.train()
    
    if args.mixup:
        print('Using Mixup')
        criterion = SoftTargetCrossEntropy()

    for key in original_weights.keys():
        encoders[key].train()
        decoders[key].train()


    for b_idx, (input_images, target_labels) in enumerate(pbar):
        with autocast(enabled=args.mixed_precision):
            
            mse_loss = 0
            decoded_weights = {}
            for key,value in original_weights.items():
                encoded_value = encoders[key](value)
                decoded_value = decoders[key](encoded_value)
                mse_loss += F.mse_loss(decoded_value, value)

                decoded_weights[key] = decoded_value
            
            update_model_weights(model_copy, decoded_weights, total_blocks=args.total_blocks, selected_layers=selected_layers, 
                                 skip_qkv=args.skip_qkv, hidden_dim=args.hidden_dim)
                 
            input_images = input_images.cuda()
            target_labels = target_labels.cuda()

            if args.mixup and (input_images.shape[0] % 2
                               == 0):  # Mixup requires even batch size
                input_images, target_labels = mixup_fn(input_images,
                                                       target_labels)
            all_pred = model_copy(input_images)
            
            prediction = all_pred
                
            # loss_ce = F.cross_entropy(prediction, target_labels)
            if len(target_labels.shape) == 1:
                loss_ce = F.cross_entropy(
                    prediction, target_labels, label_smoothing=0.1)
            else:
                loss_ce = criterion(prediction, target_labels)
            
            loss = args.mse_weight * mse_loss + args.ce_weight * loss_ce
            # loss = args.ce_weight * loss_ce
            
            if args.distillation_weight > 0:
                loss_distillation = calculate_distillation_loss(original_model, all_pred, input_images)
                # print(loss, loss_attention * args.attention_weight, loss_distillation * args.distillation_weight)
                loss = loss + args.distillation_weight * loss_distillation 
                
            loss_meter.update(loss.item())

        optimizer.zero_grad()
        if args.finetune_other_params:
            optimizer_finetune.zero_grad()

        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        if args.finetune_other_params:
            scaler.step(optimizer_finetune)

        scaler.update()
        if not args.mixup:
          #  target_labels = target_labels.argmax(1) if len(target_labels.shape) > 1 else target_labels
            acc = prediction.argmax(1).eq(target_labels).float().mean()
            acc_meter.update(acc.item())

        pbar.set_postfix(loss_current=loss.item(), loss_ravg=loss_meter.avg, accuracy=acc_meter.avg)

        if ((b_idx + 1) % args.eval_interval == 0) or (b_idx + 1 == len(train_loader)):
            model_copy.eval()
                
            acc = evaluate_new_mixed(val_loader, model_copy, -1, mp=args.mixed_precision)
            if acc > best_acc:
                best_acc = acc
                print(f"Best accuracy: {best_acc}")
                encoder_states = {key: value.state_dict() for key, value in encoders.items()}
                decoder_states = {key: value.state_dict() for key, value in decoders.items()}
                all_state = {
                    'model_state_dict': model_copy.state_dict(),
                    'best_acc': best_acc,
                    'encoder_states': encoder_states,
                    'original_weights': original_weights,
                    'selected_layers': selected_layers,
                    'decoder_states': decoder_states,
                    'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'epoch': epoch + 1,
                    'rank': args.rank,
                }

                if scheduler is not None:
                    all_state['scheduler_state'] = scheduler.state_dict()
                
                if args.finetune_other_params:
                    all_state['optimizer_finetune_state'] = optimizer_finetune.state_dict()
                    all_state['scheduler_finetune_state'] = scheduler_finetune.state_dict()

                torch.save(all_state, f'saved_models/{args.base_dir}/{args.model_name}.pth')
                           

            csv_metrics['epoch'].append(epoch + 1)
            csv_metrics['b_idx'].append(b_idx + 1)
            csv_metrics['val_acc'].append(acc)
            csv_metrics['best_acc'].append(best_acc)
            csv_metrics['train_loss'].append(loss_meter.avg)
            csv_metrics['train_acc'].append(acc_meter.avg)

            pd.DataFrame(csv_metrics).to_csv(f'saved_models/{args.base_dir}/{args.model_name}.csv', index=False)

    return best_acc


def train(encoders, decoders, original_weights, original_model, model_copy, train_loader, val_loader, 
                     optimizer, scheduler, optimizer_finetune, scheduler_finetune, selected_layers, args):
    best_acc = 0
    csv_metrics = {'epoch': [], 'b_idx':[], 'val_acc':[], 'best_acc':[], 'train_loss':[], 'train_acc':[]}#, 'is_finetuning':[]}
    # decoder.train()

    scaler = GradScaler(enabled=args.mixed_precision)
    
    if scheduler is not None:
        csv_metrics['lr'] = [scheduler.get_last_lr()[0]]

    if args.finetune_other_params:
        csv_metrics['lr_finetune'] = [scheduler_finetune.get_last_lr()[0]]
    
    for epoch in range(args.epochs):
        # best_acc = train_one_epoch(epoch, encoders, decoders, original_model, model_copy, train_loader, 
        #                            val_loader, optimizer, scheduler, scaler, args, csv_metrics, best_acc, 
        #                            optimizer_finetune=optimizer_finetune, scheduler_finetune=scheduler_finetune, selected_layers=selected_layers)
        best_acc = train_one_epoch(
                epoch, encoders, decoders, original_model, model_copy, original_weights, train_loader, 
                val_loader, optimizer, scheduler, scaler, args, csv_metrics, best_acc, 
                optimizer_finetune=optimizer_finetune, scheduler_finetune=scheduler_finetune, selected_layers=selected_layers
            )
        
        if scheduler is not None:
            scheduler.step()
            csv_metrics['lr'].append(scheduler.get_last_lr()[0])

        if args.finetune_other_params:
            scheduler_finetune.step()
            csv_metrics['lr_finetune'].append(scheduler_finetune.get_last_lr()[0])
        
    return best_acc
