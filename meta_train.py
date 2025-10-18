import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import argparse

from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from utils import *

def build_ema_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def ema_update(model, ema_state, ema_decay):
    mstate = model.state_dict()
    for k in ema_state.keys():
        if k in mstate:
            ema_state[k].mul_(ema_decay).add_(mstate[k].detach(), alpha=1.0 - ema_decay)

def swap_to_ema(model, ema_state):
    """Load EMA weights into model; return backup to restore later."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state, strict=False)
    return backup

def load_back(model, backup):
    model.load_state_dict(backup, strict=False)

# Optimizer & Scheduler
def make_optimizer_and_scheduler(model, params, stop_epoch):
    """
    AdamW + differential LR (backbone lr lower) + warmup + cosine decay
    """
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('backbone' in n) or ('encoder' in n):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': params.lr * params.backbone_lr_scale},
            {'params': head_params,     'lr': params.lr}
        ],
        weight_decay=params.weight_decay
    )

    warmup = max(1, min(params.warmup_epochs, max(1, stop_epoch // 3)))
    def lr_lambda(epoch):
        # linear warmup
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        # cosine for the rest
        progress = (epoch - warmup) / max(1, (stop_epoch - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

# Train loop
def train(params, base_loader, val_loader, model, stop_epoch):
    trlog = {
        'args': vars(params),
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'best_val_loss': float('inf'),
        'best_val_loss_epoch': -1,
        'max_acc': 0.0, 'max_acc_epoch': 0
    }

    optimizer, scheduler = make_optimizer_and_scheduler(model, params, stop_epoch)

    # EMA & gradient clipping config
    ema_state = build_ema_state(model)
    ema_decay = params.ema_decay
    patience_counter = 0

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(stop_epoch):
        t0 = time.time()
        model.train()

        avg_loss = 0.0
        total_correct = 0
        total_count = 0

        for i, (x, _) in enumerate(base_loader):
            if (i + 1) % 50 == 0:
                print(f'  - Epoch [{epoch+1}/{stop_epoch}], Episode [{i+1}/{len(base_loader)}]...')

            # đảm bảo float32 toàn tuyến để tránh xung đột Half/Float trong MPNCOV
            x = x.cuda(non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            loss, correct_this_batch = model(x)
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip_norm)
            optimizer.step()

            ema_update(model, ema_state, ema_decay)

            avg_loss += loss.item()
            total_correct += correct_this_batch.sum().item()
            total_count += x.size(0) * params.n_query

        train_loss = avg_loss / max(1, len(base_loader))
        train_acc = (total_correct / total_count) * 100 if total_count != 0 else 0.0

        # ======= Validation using EMA weights for smoother metrics =======
        model.eval()
        model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
        backup = swap_to_ema(model_to_eval, ema_state)
        with torch.no_grad():
            val_loss, val_acc = model_to_eval.test_loop(val_loader)
        load_back(model_to_eval, backup)

        # ======= Checkpointing =======
        improved = val_loss < trlog['best_val_loss']
        if improved:
            print("best (by val_loss) model! save...")
            trlog['best_val_loss'] = val_loss
            trlog['best_val_loss_epoch'] = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state}, outfile)
            patience_counter = 0
        else:
            patience_counter += 1

        # track max acc
        if val_acc > trlog['max_acc']:
            trlog['max_acc'] = val_acc
            trlog['max_acc_epoch'] = epoch

        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, f'{epoch}.tar')
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state}, outfile)

        # ======= Logging =======
        trlog['train_loss'].append(train_loss)
        trlog['train_acc'].append(train_acc)
        trlog['val_loss'].append(val_loss)
        trlog['val_acc'].append(val_acc)
        trlog['lrs'] = [[pg['lr'] for pg in optimizer.param_groups] for _ in [0]]
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        scheduler.step()

        elapsed = (time.time() - t0) / 60.0
        curr_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Epoch {epoch+1}/{stop_epoch} | {elapsed:.2f} min | LRs: {curr_lrs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc:  {trlog['max_acc']:.2f}% @ epoch {trlog['max_acc_epoch']}")
        print(f"  Best Val Loss: {trlog['best_val_loss']:.4f} @ epoch {trlog['best_val_loss_epoch']}")

        # ======= Early stopping by val_loss =======
        if params.patience > 0 and patience_counter >= params.patience:
            print(f"Early stopping (no improve in {params.patience} epochs).")
            break

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=112, type=int, choices=[112, 224], help='input image size')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate (head)')
    parser.add_argument('--epoch', default=80, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='kinetics400_mini',
                        choices=['Rareact2','kinetics400_mini','d2iving48','Rareact','k400','ucf101','hmdb51','SSv2Full','SSv2Small','tiered_imagenet','diving48'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='VideoMAES',
                        choices=['ResNet12','ResNet18','VideoMAENormal','VideoMAES','VideoMAES2','VideoMAEB','VideoMAE'])
    parser.add_argument('--tunning_mode', default='normal', choices=['normal','PSRP','SSF','ss'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc','stl','protonet'])

    parser.add_argument('--train_n_episode', default=600, type=int)
    parser.add_argument('--val_n_episode', default=600, type=int)
    parser.add_argument('--train_n_way', default=10, type=int)
    parser.add_argument('--val_n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=5, type=int)
    parser.add_argument('--n_query', default=10, type=int)
    parser.add_argument('--distributed', action='store_true', default=True)

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=200, type=int, help='used by some datasets')
    parser.add_argument('--pretrain_path', default='', help='pre-trained model .tar file path')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # ==== BDC/head ====
    parser.add_argument('--reduce_dim', default=512, type=int, help='output dim of BDC reduction layer')

    # ==== optimization extras ====
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--backbone_lr_scale', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--clip_norm', type=float, default=1.0)

    params = parser.parse_args()
    num_gpu = set_gpu(params)
    set_seed(params.seed)

    # ===== dataset routing =====
    json_file_read = False
    if params.dataset == 'Rareact':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 64
    elif params.dataset == 'kinetics400_mini':
        base_file = 'preprocessed_base.json'; val_file = 'preprocessed_val.json'; json_file_read = True; params.num_classes = 200
    elif params.dataset == 'Rareact2':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 64
    elif params.dataset in ['diving48','d2iving48']:
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 48
    elif params.dataset == 'k400':
        base_file = 'VideoMAEv2base.json'; val_file = 'VideoMAEv2val.json'; json_file_read = True; params.num_classes = 400
    elif params.dataset == 'hmdb51':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 51
    elif params.dataset == 'ucf101':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 101
    elif params.dataset == 'SSv2Full':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True
    else:
        raise ValueError('dataset error')

    # ===== dataloaders =====
    train_few_shot = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(
        params.data_path, params.image_size,
        n_query=params.n_query, n_episode=params.train_n_episode,
        json_read=json_file_read, **train_few_shot
    )
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot = dict(n_way=params.val_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(
        params.data_path, params.image_size,
        n_query=params.n_query, n_episode=params.val_n_episode,
        json_read=json_file_read, **test_few_shot
    )
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # batch shape: [n_way, n_support + n_query, dim, w, h]

    # ===== model =====
    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model], **train_few_shot)
    else:
        raise ValueError("Unknown method")

    if torch.cuda.device_count() > 1 and len(params.gpu.split(',')) > 1:
        print(f"Activating DataParallel for {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()

    # checkpoints dir
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_2TAA'
    params.checkpoint_dir += params.extra_dir
    print(params.checkpoint_dir)

    print(params.pretrain_path)
    modelfile = os.path.join(params.pretrain_path)
    model = load_model(model, modelfile)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)

    model = train(params, base_loader, val_loader, model, params.epoch)
