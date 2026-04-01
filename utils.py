import os
import math
import sys
from pathlib import Path

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
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out


# =============================================================================
# HighD graph dataset for Social-STGCNN
# =============================================================================
#
# x_nb NB_DIM layout (13 features, stored by preprocess.py):
#   [0] dx        [1] dy        [2] dvx       [3] dvy
#   [4] dax       [5] day       [6] lc_state  [7] volume
#   [8] size_bin  [9] gate      [10] I_x      [11] I_y   [12] I
#
# feature_mode → which nb indices to include
NB_FEAT_IDX = {
    'baseline':   [0, 1, 2, 3, 4, 5],
    'importance': [0, 1, 2, 3, 4, 5, 12],
    'Iy':         [0, 1, 2, 3, 4, 5, 11],
    'dim':        [0, 1, 2, 3, 4, 5, 8],
    'dimI':       [0, 1, 2, 3, 4, 5, 8, 11],
}


def get_input_size(feature_mode):
    """Total node feature channels: len(nb_feat) + 1 (is_ego flag)."""
    return len(NB_FEAT_IDX[feature_mode]) + 1


class HighDGraphDataset(Dataset):
    """Dataset for Social-STGCNN trained on highD mmap outputs.

    Reads mmap files produced by preprocess.py and converts them into
    graph tensors compatible with Social-STGCNN's train/test loop.

    Node layout per graph (V = 9 nodes):
        node 0      : ego vehicle
        node 1..8   : neighbor slots (preceding, following, leftPreceding, …)

    Feature vector per node (C = len(nb_feat_idx) + 1 channels):
        ego   : [x_rel, y_rel, vx, vy, ax, ay, 0…(pad), is_ego=1]
        neighbor : [nb_feat selected by feature_mode…, is_ego=0]
        Ego features occupy the first 6 slots; extra nb-only slots are 0 for ego.

    Returns the same 10-element tuple as TrajectoryDataset so that
    train.py / test.py require minimal changes:
        obs_traj       (V, 2, T_H)  – absolute ego+nb positions (nb = 0)
        pred_traj_gt   (V, 2, T_F)  – absolute ego future (nb = 0)
        obs_traj_rel   (V, 2, T_H)  – dummy zeros
        pred_traj_rel  (V, 2, T_F)  – dummy zeros
        non_linear_ped (V,)         – dummy zeros
        loss_mask      (V, T_H+T_F) – ones for ego, zeros for nb
        V_obs          (T_H, V, C)  – graph node features (observation)
        A_obs          (T_H, V, V)  – normalized Laplacian (observation)
        V_tr           (T_F, V, 2)  – ego future displacements (slot 0 only)
        A_tr           (T_F, V, V)  – same adjacency as last obs timestep
    """

    def __init__(self, mmap_dir, feature_mode='baseline', pred_len=15,
                 norm_lap_matr=True, indices_file=None):
        """
        Args:
            mmap_dir      : directory containing x_ego.npy, x_nb.npy, etc.
                            (output of preprocess.py)
            feature_mode  : which neighbor feature subset to use
            pred_len      : number of future timesteps to predict
            norm_lap_matr : apply normalized Laplacian to adjacency matrix
            indices_file  : path to a .npy file of integer indices that selects
                            a subset of samples (e.g. data/highD/splits/train_indices.npy).
                            None = use the full dataset.
        """
        super().__init__()
        mmap_dir = Path(mmap_dir)
        self.x_ego    = np.load(mmap_dir / 'x_ego.npy')                        # (N, T, 6)  ~178 MB → RAM
        self.x_nb     = np.load(mmap_dir / 'x_nb.npy',       mmap_mode='r')  # (N, T, K, 13) ~3 GB → mmap
        self.nb_mask  = np.load(mmap_dir / 'nb_mask.npy')                     # (N, T, K) bool ~59 MB → RAM
        self.y        = np.load(mmap_dir / 'y.npy')                           # (N, Tf, 2)  ~99 MB → RAM
        self.x_last   = np.load(mmap_dir / 'x_last_abs.npy')                  # (N, 2)       ~7 MB → RAM

        assert feature_mode in NB_FEAT_IDX, \
            f"feature_mode must be one of {list(NB_FEAT_IDX)}, got '{feature_mode}'"

        # indices subset (None → all)
        if indices_file is not None:
            self.indices = np.load(indices_file).astype(np.int64)
        else:
            self.indices = None

        self.nb_idx   = NB_FEAT_IDX[feature_mode]
        self.nb_dim   = len(self.nb_idx)
        self.C        = self.nb_dim + 1        # unified channel count (nb_dim + is_ego)
        self.T_H      = self.x_ego.shape[1]
        self.T_F      = min(pred_len, self.y.shape[1])
        self.K        = self.x_nb.shape[2]    # 8 neighbor slots
        self.V        = 1 + self.K            # 9 nodes total
        self.norm_lap = norm_lap_matr

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.x_ego.shape[0]

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        x_ego    = np.array(self.x_ego[idx],    np.float32)   # (T_H, 6)
        x_nb     = np.array(self.x_nb[idx],     np.float32)   # (T_H, K, 13)
        nb_mask  = np.array(self.nb_mask[idx],  bool)         # (T_H, K)
        y        = np.array(self.y[idx],        np.float32)   # (Tf, 2)
        norm_ctr = np.array(self.x_last[idx],   np.float32)   # (2,)

        # ── V_obs: (T_H, V, C) ────────────────────────────────────────────────
        V_obs = np.zeros((self.T_H, self.V, self.C), np.float32)

        # ego node (index 0): first 6 channels = x_ego features; last channel = is_ego=1
        V_obs[:, 0, :6]         = x_ego
        V_obs[:, 0, self.C - 1] = 1.0

        # neighbor nodes (index 1..K): selected nb features; last channel = is_ego=0
        nb_sel = x_nb[:, :, self.nb_idx]   # (T_H, K, nb_dim)
        V_obs[:, 1:, :self.nb_dim] = nb_sel

        # zero out absent neighbor slots
        absent = ~nb_mask                  # (T_H, K)
        V_obs[:, 1:, :][absent] = 0.0

        # ── A_obs: (T_H, V, V) normalized Laplacian ──────────────────────────
        A_obs = self._build_adj(nb_mask)   # (T_H, V, V)

        # ── V_tr: (T_F, V, 2) — ego future relative displacements ────────────
        # y[t] = absolute future pos relative to norm_ctr (ego's last obs pos)
        # displacement at step t: y[t] - y[t-1]  (y[-1] ≡ 0 since norm_ctr is ref)
        y_pad = np.concatenate([np.zeros((1, 2), np.float32), y[:self.T_F]], axis=0)
        ego_disp = y_pad[1:] - y_pad[:-1]  # (T_F, 2)

        V_tr = np.zeros((self.T_F, self.V, 2), np.float32)
        V_tr[:, 0, :] = ego_disp           # only ego slot populated

        # ── A_tr: (T_F, V, V) — replicate last obs adjacency ─────────────────
        A_tr = np.broadcast_to(A_obs[-1:], (self.T_F, self.V, self.V)).copy()

        # ── auxiliary tensors for test.py metrics ────────────────────────────
        # obs_traj (V, 2, T_H): absolute positions; only ego is valid
        obs_traj = np.zeros((self.V, 2, self.T_H), np.float32)
        obs_traj[0, :, :] = (x_ego[:, :2] + norm_ctr).T   # (2, T_H)

        # pred_traj_gt (V, 2, T_F): absolute future; only ego is valid
        pred_traj_gt = np.zeros((self.V, 2, self.T_F), np.float32)
        pred_traj_gt[0, :, :] = (y[:self.T_F] + norm_ctr).T  # (2, T_F)

        loss_mask = np.zeros((self.V, self.T_H + self.T_F), np.float32)
        loss_mask[0, :] = 1.0              # supervise ego only

        return [
            torch.from_numpy(obs_traj),                                    # (V, 2, T_H)
            torch.from_numpy(pred_traj_gt),                                # (V, 2, T_F)
            torch.zeros(self.V, 2, self.T_H),                             # obs_traj_rel (dummy)
            torch.zeros(self.V, 2, self.T_F),                             # pred_traj_gt_rel (dummy)
            torch.zeros(self.V),                                           # non_linear_ped (dummy)
            torch.from_numpy(loss_mask),                                   # (V, T_H+T_F)
            torch.from_numpy(V_obs),                                       # (T_H, V, C)
            torch.from_numpy(A_obs),                                       # (T_H, V, V)
            torch.from_numpy(V_tr),                                        # (T_F, V, 2)
            torch.from_numpy(A_tr),                                        # (T_F, V, V)
        ]

    def _build_adj(self, nb_mask):
        """Build normalized Laplacian adjacency per timestep.

        Binary structure: ego (0) is connected to every present neighbor.
        Self-loops included before normalization.
        nb_mask: (T_H, K) bool
        Returns: (T_H, V, V) float32
        """
        # adj: (T_H, V, V) — start with identity (self-loops)
        adj = np.eye(self.V, dtype=np.float64)[None].repeat(self.T_H, axis=0)
        for k in range(self.K):
            present = nb_mask[:, k]          # (T_H,) bool
            adj[present, 0, k + 1] = 1.0
            adj[present, k + 1, 0] = 1.0

        if self.norm_lap:
            degree = adj.sum(axis=2)         # (T_H, V)
            d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
            eye = np.eye(self.V, dtype=np.float64)[None]
            L = eye - d_inv_sqrt[:, :, None] * adj * d_inv_sqrt[:, None, :]
            return L.astype(np.float32)
        else:
            return adj.astype(np.float32)
