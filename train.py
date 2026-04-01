

import os

import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import *
from metrics import *
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *
from config_utils import load_config_and_parse
from tqdm import tqdm

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2,
                    help='Node feature channels. Auto-set from --feature_mode when using highD.')
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')

# HighD-specific parameters
parser.add_argument('--highd', action='store_true', default=False,
                    help='Use HighDGraphDataset instead of TrajectoryDataset')
parser.add_argument('--mmap_dir', type=str, default='data/highD/mmap',
                    help='Path to mmap directory produced by preprocess.py')
parser.add_argument('--splits_dir', type=str, default='data/highD/splits',
                    help='Directory containing train/val/test index .npy files '
                         'produced by split.py (e.g. train_indices.npy)')
parser.add_argument('--feature_mode', type=str, default='baseline',
                    choices=['baseline', 'importance', 'Iy', 'dim', 'dimI'],
                    help='Neighbor feature subset for highD training')

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_workers', type=int, default=64,
                    help='number of DataLoader worker processes')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for reproducibility')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
                    
args = load_config_and_parse(parser)

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("[INFO] Training initiating....")
print(args)


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len

if args.highd:
    from pathlib import Path
    input_feat   = get_input_size(args.feature_mode)
    splits_dir   = Path(args.splits_dir)
    dset_train = HighDGraphDataset(
            args.mmap_dir,
            feature_mode=args.feature_mode,
            pred_len=pred_seq_len,
            norm_lap_matr=True,
            indices_file=splits_dir / 'train_indices.npy')
    dset_val = HighDGraphDataset(
            args.mmap_dir,
            feature_mode=args.feature_mode,
            pred_len=pred_seq_len,
            norm_lap_matr=True,
            indices_file=splits_dir / 'val_indices.npy')
else:
    input_feat = args.input_size
    data_set = './datasets/'+args.dataset+'/'
    dset_train = TrajectoryDataset(
            data_set+'train/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=True)
    dset_val = TrajectoryDataset(
            data_set+'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=True)

loader_batch_size = args.batch_size if args.highd else 1
loader_train = DataLoader(
        dset_train,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=args.num_workers if args.highd else 0,
        pin_memory=True)

loader_val = DataLoader(
        dset_val,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.highd else 1,
        pin_memory=True)


#Defining the model

model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
input_feat=input_feat,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()


#Training settings 

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    


checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    


print('[INFO] Data and model loaded')
print('[INFO] Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

def train(epoch):
    global metrics,loader_train
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    accum_size = 1 if args.highd else args.batch_size
    turn_point =int(loader_len/accum_size)*accum_size+ loader_len%accum_size -1


    pbar = tqdm(enumerate(loader_train), total=loader_len, desc=f'Train {epoch}', leave=False)
    for cnt,batch in pbar:
        batch_count+=1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch



        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())

        V_pred = V_pred.permute(0,2,3,1)



        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        # highD: supervise ego node only (index 0); pedestrian dataset: all nodes
        if args.highd:
            _V_pred = V_pred[:, 0:1, :] if V_pred.dim() == 3 else V_pred[:, :, 0:1, :]
            _V_tr   = V_tr[:, 0:1, :]   if V_tr.dim()   == 3 else V_tr[:, :, 0:1, :]
            if _V_pred.dim() == 4:
                _V_pred = _V_pred.flatten(0, 1)
                _V_tr   = _V_tr.flatten(0, 1)
        else:
            _V_pred, _V_tr = V_pred, V_tr

        l = graph_loss(_V_pred, _V_tr)

        if accum_size > 1 and batch_count % accum_size != 0 and cnt != turn_point:
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = l if accum_size == 1 else loss / accum_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            loss_batch += loss.item()
            pbar.set_postfix({'loss': f'{loss_batch/batch_count:.4f}'})

    metrics['train_loss'].append(loss_batch/batch_count)
    print(f'[EPOCH {epoch:02d}] TRAIN Loss: {loss_batch/batch_count:.4f}')
    



def vald(epoch):
    global metrics,loader_val,constant_metrics
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    accum_size = 1 if args.highd else args.batch_size
    turn_point =int(loader_len/accum_size)*accum_size+ loader_len%accum_size -1
    
    pbar = tqdm(enumerate(loader_val), total=loader_len, desc=f'Val   {epoch}', leave=False)
    for cnt,batch in pbar:
        batch_count+=1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())

        V_pred = V_pred.permute(0,2,3,1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if args.highd:
            _V_pred = V_pred[:, 0:1, :] if V_pred.dim() == 3 else V_pred[:, :, 0:1, :]
            _V_tr   = V_tr[:, 0:1, :]   if V_tr.dim()   == 3 else V_tr[:, :, 0:1, :]
            if _V_pred.dim() == 4:
                _V_pred = _V_pred.flatten(0, 1)
                _V_tr   = _V_tr.flatten(0, 1)
        else:
            _V_pred, _V_tr = V_pred, V_tr

        l = graph_loss(_V_pred, _V_tr)

        if accum_size > 1 and batch_count % accum_size != 0 and cnt != turn_point:
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = l if accum_size == 1 else loss / accum_size
            is_fst_loss = True
            loss_batch += loss.item()
            pbar.set_postfix({'loss': f'{loss_batch/batch_count:.4f}'})

    metrics['val_loss'].append(loss_batch/batch_count)
    print(f'[EPOCH {epoch:02d}] VALID Loss: {loss_batch/batch_count:.4f}')
    
    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK
        print(f"⭐ Best Model Saved (Val Loss: {metrics['val_loss'][-1]:.4f})")


print('[INFO] Training started ...')
for epoch in range(args.num_epochs):
    print(f'\n==================== Epoch {epoch:02d} ====================')
    train(epoch)
    vald(epoch)
    if args.use_lrschd:
        scheduler.step()
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
    
    with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)  




