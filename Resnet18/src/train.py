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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Using device:", device)

    # 数据迭代器
    dl = MyDataLoader()
    # 如果启用 --train_full 则使用全部训练样本并跳过验证
    if args.train_full:
        train_set_name = 'all'
        val_loader = None
    else:
        train_set_name = 'train'

    train_loader = dl.load_multi_modal_dataiter(set=train_set_name,
                                                frames_per_clip=args.frames_per_clip,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers)
    if not args.train_full:
        val_loader = dl.load_multi_modal_dataiter(set='val',
                                                  frames_per_clip=args.frames_per_clip,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers)

    # 模型
    model = VideoRecognitionModel(num_classes=args.num_classes,
                                  num_frames=args.frames_per_clip,
                                  lstm_hidden_size=args.lstm_hidden_size,
                                  lstm_num_layers=args.lstm_num_layers)
    model = model.to(device)

    # 损失/优化器/调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--frames_per_clip', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)  # Windows / notebook 默认 0 更安全
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_step', type=int, default=7)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--train_full', action='store_true') # 决定是否使用整个训练集训练（不必要验证）
    args = parser.parse_args()
    main(args)

'''
Epoch 13/20 | Time 1015.4s | Train loss 1.1627 acc 0.7686 | Val loss 0.9814 acc 0.7812
Epoch 14/20 | Time 906.8s | Train loss 1.1123 acc 0.7739 | Val loss 0.9331 acc 0.7812
Epoch 15/20 | Time 948.0s | Train loss 1.0277 acc 0.7899 | Val loss 0.8776 acc 0.8516
Epoch 16/20 | Time 968.2s | Train loss 0.9548 acc 0.8378 | Val loss 0.8586 acc 0.8125
Epoch 17/20 | Time 1156.7s | Train loss 0.9294 acc 0.8245 | Val loss 0.8240 acc 0.8359
Epoch 15/20 | Time 948.0s | Train loss 1.0277 acc 0.7899 | Val loss 0.8776 acc 0.8516
Epoch 16/20 | Time 968.2s | Train loss 0.9548 acc 0.8378 | Val loss 0.8586 acc 0.8125
Epoch 17/20 | Time 1156.7s | Train loss 0.9294 acc 0.8245 | Val loss 0.8240 acc 0.8359
Epoch 18/20 | Time 1142.0s | Train loss 0.8838 acc 0.8475 | Val loss 0.8290 acc 0.8203
Epoch 17/20 | Time 1156.7s | Train loss 0.9294 acc 0.8245 | Val loss 0.8240 acc 0.8359
Epoch 18/20 | Time 1142.0s | Train loss 0.8838 acc 0.8475 | Val loss 0.8290 acc 0.8203
Epoch 18/20 | Time 1142.0s | Train loss 0.8838 acc 0.8475 | Val loss 0.8290 acc 0.8203
Epoch 19/20 | Time 1082.8s | Train loss 0.8814 acc 0.8209 | Val loss 0.8047 acc 0.8047
Epoch 20/20 | Time 1180.7s | Train loss 0.8114 acc 0.8537 | Val loss 0.7668 acc 0.8203
Training finished. Best val acc: 0.8515625
'''

'''
Using device: cuda
Epoch 1/20 | Time 370.6s | Train loss 2.8874 acc 0.1726
Epoch 2/20 | Time 367.0s | Train loss 2.5554 acc 0.3492
Epoch 3/20 | Time 370.3s | Train loss 2.2256 acc 0.5238
Epoch 4/20 | Time 366.3s | Train loss 1.9231 acc 0.5754
Epoch 5/20 | Time 365.4s | Train loss 1.6657 acc 0.6746
Epoch 6/20 | Time 370.1s | Train loss 1.4576 acc 0.7004
Epoch 7/20 | Time 366.2s | Train loss 1.3051 acc 0.7242
Epoch 8/20 | Time 370.0s | Train loss 1.1202 acc 0.8036
Epoch 9/20 | Time 374.3s | Train loss 1.0123 acc 0.8075
Epoch 10/20 | Time 371.5s | Train loss 0.9117 acc 0.8492
Epoch 11/20 | Time 367.5s | Train loss 0.8355 acc 0.8472
Epoch 12/20 | Time 371.3s | Train loss 0.7490 acc 0.8611
Epoch 13/20 | Time 368.0s | Train loss 0.7167 acc 0.8651
Epoch 14/20 | Time 370.6s | Train loss 0.6741 acc 0.8889
Epoch 15/20 | Time 369.4s | Train loss 0.6113 acc 0.9107
Epoch 16/20 | Time 364.8s | Train loss 0.5736 acc 0.9147
Epoch 17/20 | Time 363.0s | Train loss 0.5378 acc 0.9286
Epoch 18/20 | Time 367.4s | Train loss 0.5141 acc 0.9286
Epoch 19/20 | Time 362.3s | Train loss 0.4817 acc 0.9266
Epoch 20/20 | Time 363.0s | Train loss 0.4615 acc 0.9365
Training finished. Best train loss: 0.4615383803371399
'''