import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import argparse

from data.datamgr import SetDataManager
from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from utils import *

def build_ema_state(model):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state

@torch.no_grad()
def ema_update(model, ema_state, ema_decay):
    model_state = model.state_dict()
    for k in ema_state.keys():
        if k in model_state:
            ema_state[k].mul_(ema_decay).add_(model_state[k].detach(), alpha=1.0 - ema_decay)

def swap_to_ema(model, ema_state):
    """Load EMA weights into model, return a backup of current weights."""
    current = {k: v.detach().clone() for k, v in model.state_dict().items()}
    missing, unexpected = model.load_state_dict(ema_state, strict=False)
    return current

def load_back(model, backup_state):
    model.load_state_dict(backup_state, strict=False)


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

    warmup_epochs = max(1, params.warmup_epochs)
    warmup_epochs = min(warmup_epochs, stop_epoch // 3)  # an toàn

    def lr_lambda(epoch):
        # linear warmup
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # cosine decay for the rest
        progress = (epoch - warmup_epochs) / max(1, (stop_epoch - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def train(params, base_loader, val_loader, model, stop_epoch):
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'], trlog['val_loss'], trlog['train_acc'], trlog['val_acc'] = [], [], [], []
    trlog['best_val_loss'], trlog['best_val_loss_epoch'] = float('inf'), -1
    trlog['max_acc'], trlog['max_acc_epoch'] = 0.0, 0

    # Optimizer + Scheduler (AdamW + warmup + cosine)
    optimizer, scheduler = make_optimizer_and_scheduler(model, params, stop_epoch)

    # AMP / Grad clip / EMA
    scaler = torch.cuda.amp.GradScaler(enabled=params.use_amp)
    ema_state = build_ema_state(model)
    ema_decay = params.ema_decay

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    no_improve_epochs = 0

    for epoch in range(stop_epoch):
        start_time = time.time()
        model.train()

        avg_loss = 0.0
        total_correct = 0
        total_count = 0

        for i, (x, _) in enumerate(base_loader):
            if (i + 1) % 50 == 0:  # log mỗi 50 episode
                print(f'  - Epoch [{epoch+1}/{stop_epoch}], Episode [{i+1}/{len(base_loader)}]...')

            x = x.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=params.use_amp):
                loss, correct_this_batch = model(x)
                loss = loss.mean()

            scaler.scale(loss).backward()

            # gradient clipping để ổn định
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # EMA update
            ema_update(model, ema_state, ema_decay)

            avg_loss += loss.item()
            total_correct += correct_this_batch.sum().item()
            total_count += x.size(0) * params.n_query

        train_loss = avg_loss / max(1, len(base_loader))
        train_acc = (total_correct / total_count) * 100 if total_count != 0 else 0

        model.eval()
        model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
        backup_state = swap_to_ema(model_to_eval, ema_state)  # load EMA
        with torch.no_grad():
            val_loss, val_acc = model_to_eval.test_loop(val_loader)
        load_back(model_to_eval, backup_state)  # khôi phục

        # ---- Best by val_loss (ổn định hơn acc) ----
        if val_loss < trlog['best_val_loss']:
            print("best (by val_loss) model! save...")
            trlog['best_val_loss'], trlog['best_val_loss_epoch'] = val_loss, epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state_to_save}, outfile)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # vẫn lưu max acc để theo dõi
        if val_acc > trlog['max_acc']:
            trlog['max_acc'], trlog['max_acc_epoch'] = val_acc, epoch

        # save theo chu kỳ
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, f'{epoch}.tar')
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state_to_save}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'epoch': epoch, 'state': state_to_save}, outfile)

        # log
        trlog['train_loss'].append(train_loss)
        trlog['train_acc'].append(train_acc)
        trlog['val_loss'].append(val_loss)
        trlog['val_acc'].append(val_acc)
        trlog['lrs'] = [[pg['lr'] for pg in optimizer.param_groups] for _ in [0]]  # lưu LR hiện tại (đơn giản)
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        # step scheduler cuối epoch
        scheduler.step()

        # console summary
        elapsed = (time.time() - start_time) / 60.0
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Epoch {epoch+1}/{stop_epoch} | {elapsed:.2f} min | LRs: {current_lrs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Best Val Acc: {trlog['max_acc']:.2f}% at epoch {trlog['max_acc_epoch']}")
        print(f"  Best Val Loss: {trlog['best_val_loss']:.4f} at epoch {trlog['best_val_loss_epoch']}")

        # Early stopping nếu không cải thiện theo val_loss
        if params.patience > 0 and no_improve_epochs >= params.patience:
            print(f"Early stopping (no improve in {params.patience} epochs).")
            break

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===== các tham số gốc =====
    parser.add_argument('--image_size', default=224, type=int, choices=[112, 224], help='input image size')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate (head)')
    parser.add_argument('--epoch', default=80, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet',
                        choices=['Rareact2','kinetics400_mini','d2iving48','Rareact','k400','ucf101','hmdb51','SSv2Full','SSv2Small','tiered_imagenet','diving48'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12',
                        choices=['ResNet12', 'ResNet18', 'VideoMAENormal','VideoMAES','VideoMAES2','VideoMAEB','VideoMAE'])
    parser.add_argument('--tunning_mode', default='normal', choices=['normal', 'PSRP', 'SSF','ss'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'stl','protonet'])

    parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=600, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=10, type=int, help='classes per episode (train)')
    parser.add_argument('--val_n_way', default=5, type=int, help='classes per episode (val)')
    parser.add_argument('--n_shot', default=1, type=int, help='support samples per class')
    parser.add_argument('--n_query', default=10, type=int, help='query samples per class')
    parser.add_argument('--distributed', action='store_true', default=True)

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
    parser.add_argument('--pretrain_path', default='', help='pre-trained model .tar file path')
    parser.add_argument('--save_freq', default=10, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    # ===== BDC/head =====
    parser.add_argument('--reduce_dim', default=512, type=int, help='output dim of BDC reduction layer')

    # ===== các tham số mới để cải thiện huấn luyện =====
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='AdamW weight decay')
    parser.add_argument('--backbone_lr_scale', type=float, default=0.1, help='LR scale for backbone vs head')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='linear warmup epochs before cosine')
    parser.add_argument('--patience', type=int, default=8, help='early stopping by val_loss (0 to disable)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='mixed precision training')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay for weights')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='gradient clipping max norm')

    params = parser.parse_args()
    num_gpu = set_gpu(params)

    set_seed(params.seed)

    json_file_read = False
    if params.dataset == 'Rareact':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 64
    elif params.dataset == 'kinetics400_mini':
        base_file = 'preprocessed_base.json'; val_file = 'preprocessed_val.json'; json_file_read = True; params.num_classes = 200
    elif params.dataset == 'Rareact2':
        base_file = 'base.json'; val_file = 'val.json'; json_file_read = True; params.num_classes = 64
    elif params.dataset == 'diving48' or params.dataset == 'd2iving48':
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
        ValueError('dataset error')

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                  n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query,
                                 n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot_params)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model], **train_few_shot_params)
    else:
        raise ValueError("Unknown method")

    if torch.cuda.device_count() > 1 and len(params.gpu.split(',')) > 1:
        print(f"Activating DataParallel for {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()

    # model save path
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
