# File: Resnet18/src/train.py
import os
import time
import torch
import argparse
import numpy as np
from torch import nn
from torch import optim
from torch.cuda import amp
from utils.data_loader import DataLoader as MyDataLoader
from models.lstm import VideoRecognitionModel

def accuracy(preds, labels):
    preds_cls = preds.argmax(dim=1)
    return (preds_cls == labels).float().mean().item()

def save_checkpoint(state, path):
    torch.save(state, path)

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for batch in loader:
        rgb = batch['rgb'].to(device)           # (B, T, C, H, W)
        depth = batch['depth'].to(device)
        infrared = batch['infrared'].to(device)
        lengths = batch['lengths']              # CPU tensor (B,)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast():
                logits = model(rgb, depth, infrared, lengths=lengths)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(rgb, depth, infrared, lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_acc = accuracy(logits.detach().cpu(), labels.detach().cpu())
        running_loss += loss.item()
        running_acc += batch_acc
        n_batches += 1

    return running_loss / max(1, n_batches), running_acc / max(1, n_batches)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            infrared = batch['infrared'].to(device)
            lengths = batch['lengths']
            labels = batch['labels'].to(device)

            logits = model(rgb, depth, infrared, lengths=lengths)
            loss = criterion(logits, labels)

            batch_acc = accuracy(logits.cpu(), labels.cpu())
            running_loss += loss.item()
            running_acc += batch_acc
            n_batches += 1

    return running_loss / max(1, n_batches), running_acc / max(1, n_batches)

def load_dataiter(dl, train_full, frames_per_clip, sampling, batch_size, shuffle, num_workers):
    if train_full:
        return dl.load_multi_modal_dataiter(set='all',
                                            frames_per_clip=frames_per_clip,
                                            sampling=sampling,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers), None
    else:
        return dl.load_multi_modal_dataiter(set='train',
                                            frames_per_clip=frames_per_clip,
                                            sampling=sampling,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,), \
                dl.load_multi_modal_dataiter(set='val', 
                                            frames_per_clip=frames_per_clip,
                                            sampling=sampling,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)

def resume_training(ckpt_path, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # 数据迭代器
    dl = MyDataLoader()
    train_loader, val_loader = load_dataiter(dl, args.train_full, args.frames_per_clip, args.sampling, 
                                            args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 模型
    modalities = ['rgb'] if args.only_rgb else ['rgb', 'depth', 'infrared']
    model = VideoRecognitionModel(num_classes=args.num_classes,
                                  num_frames=args.frames_per_clip,
                                  modalities=modalities,
                                  lstm_hidden_size=args.lstm_hidden_size,
                                  lstm_num_layers=args.lstm_num_layers,
                                  learn_weights=args.learn_weights,
                                  use_lstm=not args.no_lstm)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # 损失/优化器/调度
    criterion = nn.CrossEntropyLoss()
    
    # 设置不同的学习率组：ResNet主干使用较低的学习率
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:  # ResNet参数
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': other_params, 'lr': args.lr}
    ]
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    optimizer.load_state_dict(checkpoint['optim_state'])
    
    # 选择调度器
    if args.scheduler_type == 'plateau' and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if not args.train_full else 'min', 
                                                         factor=args.lr_factor, patience=args.patience, 
                                                         min_lr=args.min_lr, verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    # 混合精度支持
    use_amp = args.use_amp and (device.type == 'cuda')
    scaler = amp.GradScaler() if use_amp else None

    # 恢复最佳指标
    if args.train_full:
        best_metric = checkpoint.get('best_metric', float('inf'))
    else:
        best_metric = checkpoint.get('best_metric', 0.0)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 从下一个epoch开始训练
    start_epoch = checkpoint['epoch'] + 1
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip=args.grad_clip)
        # 如果有验证集则评估，否则跳过验证
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = None, None

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if val_loader is not None:
                scheduler.step(val_acc if not args.train_full else val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        if val_loader is not None:
            print(f"Epoch {epoch}/{args.epochs} | Time {elapsed:.1f}s | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} | Time {elapsed:.1f}s | Train loss {train_loss:.4f} acc {train_acc:.4f}")

        # 保存 checkpoint：如果使用验证集按 val_acc 判断，否则按 train_loss 判断（越小越好）
        if val_loader is not None:
            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc
        else:
            is_best = train_loss < best_metric
            if is_best:
                best_metric = train_loss

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_metric': best_metric
        }
        save_checkpoint(ckpt, os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pth"))
        if is_best:
            save_checkpoint(ckpt, os.path.join(args.ckpt_dir, "best.pth"))

    if val_loader is not None:
        print("Resume training finished. Best val acc:", best_metric)
    else:
        print("Resume training finished. Best train loss:", best_metric)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # 数据迭代器
    dl = MyDataLoader()
    # 如果启用 --train_full 则使用全部训练样本并跳过验证
    train_loader, val_loader = load_dataiter(dl, args.train_full, args.frames_per_clip, args.sampling, 
                                            args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 模型
    modalities = ['rgb'] if args.only_rgb else ['rgb', 'depth', 'infrared']
    model = VideoRecognitionModel(num_classes=args.num_classes,
                                  num_frames=args.frames_per_clip,
                                  modalities=modalities,
                                  lstm_hidden_size=args.lstm_hidden_size,
                                  lstm_num_layers=args.lstm_num_layers,
                                  learn_weights=args.learn_weights,
                                  use_lstm=not args.no_lstm)
    model = model.to(device)

    # 损失/优化器/调度
    criterion = nn.CrossEntropyLoss()
    
    # 设置不同的学习率组：ResNet主干使用较低的学习率
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:  # ResNet参数
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': other_params, 'lr': args.lr}
    ]
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    
    # 选择调度器
    if args.scheduler_type == 'plateau' and val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if not args.train_full else 'min', 
                                                         factor=args.lr_factor, patience=args.patience, 
                                                         min_lr=args.min_lr, verbose=True)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # 混合精度支持
    use_amp = args.use_amp and (device.type == 'cuda')
    scaler = amp.GradScaler() if use_amp else None

    # 如果使用验证集，则按验证 accuracy 判定最佳模型；否则按训练 loss（越小越好）判定
    if args.train_full:
        best_metric = float('inf')  # 用训练 loss，越小越好
    else:
        best_metric = 0.0  # 用验证 acc，越大越好
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip=args.grad_clip)
        # 如果有验证集则评估，否则跳过验证
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = None, None

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if val_loader is not None:
                scheduler.step(val_acc if not args.train_full else val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        if val_loader is not None:
            print(f"Epoch {epoch}/{args.epochs} | Time {elapsed:.1f}s | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} | Time {elapsed:.1f}s | Train loss {train_loss:.4f} acc {train_acc:.4f}")

        # 保存 checkpoint：如果使用验证集按 val_acc 判断，否则按 train_loss 判断（越小越好）
        if val_loader is not None:
            is_best = val_acc > best_metric
            if is_best:
                best_metric = val_acc
        else:
            is_best = train_loss < best_metric
            if is_best:
                best_metric = train_loss

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_metric': best_metric
        }
        save_checkpoint(ckpt, os.path.join(args.ckpt_dir, f"ckpt_epoch{epoch}.pth"))
        if is_best:
            save_checkpoint(ckpt, os.path.join(args.ckpt_dir, "best.pth"))

    if val_loader is not None:
        print("Training finished. Best val acc:", best_metric)
    else:
        print("Training finished. Best train loss:", best_metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frames_per_clip', type=int, default=64)
    parser.add_argument('--sampling', type=str, default='uniform', choices=['all', 'uniform', 'random'],
                        help='Frame sampling strategy: "uniform" or "random"')
    parser.add_argument('--num_workers', type=int, default=0)  # Windows / notebook 默认 0 更安全
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                        help='Learning rate for ResNet backbone (when unfrozen)')
    parser.add_argument('--weight_decay', type=float, default=0.0) # L2 正则化系数，暂时设为 0 XXX
    parser.add_argument('--lr_step', type=int, default=7)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_type', type=str, default='step', choices=['step', 'plateau'],
                        help='Learning rate scheduler type: "step" or "plateau"')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='Factor by which the learning rate will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate (for ReduceLROnPlateau)')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--train_full', action='store_true',
                        help='If set, use the full training set and skip validation')
    parser.add_argument('--only_rgb', action='store_true',
                        help='If set, use only RGB modality')
    parser.add_argument('--learn_weights', action='store_true',
                        help='Whether to learn modality concatenation weights during training')
    parser.add_argument('--no_lstm', action='store_true', default=False,
                        help='Whether to use LSTM for temporal modeling (default: False). If True, use average pooling.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    if args.resume:
        resume_training(args.resume, args)
    else:
        main(args)