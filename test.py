import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import social_stgcnn
from config_utils import load_config_and_parse
from tqdm import tqdm
import copy

def test(KSTEPS=20):
    global loader_test, model, args
    model.eval()
    ade_bigls  = []
    fde_bigls  = []
    rmse_bigls = []
    raw_data_dict = {}
    step = 0

    pbar = tqdm(loader_test, desc='Test', leave=True)
    with torch.no_grad():
        for batch in pbar:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
                loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs)
            V_pred = V_pred.permute(0, 2, 3, 1)          # (B, T_F, V, 5)

            if args.highd:
                # ── Fully vectorized over entire batch ───────────────────────
                # vp: (B, T_F, 1, 5),  vt: (B, T_F, 1, 2)
                vp = V_pred[:, :, 0:1, :]
                vt = V_tr[:, :, 0:1, :].cpu().numpy()

                # init_pos: (B, 1, 1, 2) — ego's last observed absolute position
                init_pos = obs_traj[:, 0, :, -1].cpu().numpy()[:, np.newaxis, np.newaxis, :]

                # absolute ground truth & mean prediction: (B, T_F, 1, 2)
                V_y_abs   = np.cumsum(vt, axis=1) + init_pos
                mean_abs  = np.cumsum(vp[:, :, :, 0:2].cpu().numpy(), axis=1) + init_pos

                # RMSE: sqrt( mean_t( ||mean_abs - gt||^2 ) ) per sample
                rmse_dist = np.sqrt(np.sum((mean_abs - V_y_abs) ** 2, axis=-1))  # (B, T_F, 1)
                rmse_per  = np.sqrt(rmse_dist.mean(axis=1)).squeeze(-1)           # (B,)

                # Batched bivariate Gaussian: (B, T_F, 1, *)
                sx   = torch.exp(vp[:, :, :, 2])
                sy   = torch.exp(vp[:, :, :, 3])
                corr = torch.tanh(vp[:, :, :, 4])
                cov  = torch.zeros(*vp.shape[:3], 2, 2, device=vp.device)
                cov[:, :, :, 0, 0] = sx * sx
                cov[:, :, :, 0, 1] = corr * sx * sy
                cov[:, :, :, 1, 0] = corr * sx * sy
                cov[:, :, :, 1, 1] = sy * sy
                mvnormal = torchdist.MultivariateNormal(vp[:, :, :, 0:2], cov)

                # Sample KSTEPS times; track min-ADE and min-FDE per sample
                B = vp.shape[0]
                ade_all = np.full((KSTEPS, B), np.inf)
                fde_all = np.full((KSTEPS, B), np.inf)
                for k in range(KSTEPS):
                    samp     = mvnormal.sample().cpu().numpy()             # (B, T_F, 1, 2)
                    samp_abs = np.cumsum(samp, axis=1) + init_pos          # (B, T_F, 1, 2)
                    dist     = np.sqrt(np.sum((samp_abs - V_y_abs) ** 2, axis=-1))  # (B, T_F, 1)
                    ade_all[k] = dist.mean(axis=1).squeeze(-1)
                    fde_all[k] = dist[:, -1, :].squeeze(-1)

                ade_bigls.extend(ade_all.min(axis=0).tolist())
                fde_bigls.extend(fde_all.min(axis=0).tolist())
                rmse_bigls.extend(rmse_per.tolist())

            else:
                # ── pedestrian: batch_size=1, per-sample processing ──────────
                b = 0
                step += 1
                num_of_objs = obs_traj_rel.shape[1]
                vp = V_pred[b].squeeze()[:, :num_of_objs, :]   # (T_F, V, 5)
                vt = V_tr[b].squeeze()[:, :num_of_objs, :]     # (T_F, V, 2)

                V_x = seq_to_nodes(obs_traj[b:b+1].data.cpu().numpy().copy())
                V_y_rel_to_abs = nodes_rel_to_nodes_abs(
                    vt.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
                mean_pred_abs = nodes_rel_to_nodes_abs(
                    vp[:, :, 0:2].cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())

                raw_data_dict[step] = {'trgt': copy.deepcopy(V_y_rel_to_abs), 'pred': []}

                ade_ls = {n: [] for n in range(num_of_objs)}
                fde_ls = {n: [] for n in range(num_of_objs)}

                sx   = torch.exp(vp[:, :, 2])
                sy   = torch.exp(vp[:, :, 3])
                corr = torch.tanh(vp[:, :, 4])
                cov  = torch.zeros(vp.shape[0], vp.shape[1], 2, 2).cuda()
                cov[:, :, 0, 0] = sx * sx
                cov[:, :, 0, 1] = corr * sx * sy
                cov[:, :, 1, 0] = corr * sx * sy
                cov[:, :, 1, 1] = sy * sy
                mvnormal = torchdist.MultivariateNormal(vp[:, :, 0:2], cov)

                for k in range(KSTEPS):
                    V_pred_sample = mvnormal.sample()
                    V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                        V_pred_sample.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
                    raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
                    for n in range(num_of_objs):
                        ade_ls[n].append(ade([V_pred_rel_to_abs[:, n:n+1, :]], [V_y_rel_to_abs[:, n:n+1, :]], [1]))
                        fde_ls[n].append(fde([V_pred_rel_to_abs[:, n:n+1, :]], [V_y_rel_to_abs[:, n:n+1, :]], [1]))

                for n in range(num_of_objs):
                    ade_bigls.append(min(ade_ls[n]))
                    fde_bigls.append(min(fde_ls[n]))
                    rmse_bigls.append(np.sqrt(np.mean(
                        np.sum((mean_pred_abs[:, n, :] - V_y_rel_to_abs[:, n, :]) ** 2, axis=1))))

    ade_  = sum(ade_bigls)  / len(ade_bigls)
    fde_  = sum(fde_bigls)  / len(fde_bigls)
    rmse_ = sum(rmse_bigls) / len(rmse_bigls)
    return ade_, fde_, rmse_, raw_data_dict


test_parser = argparse.ArgumentParser()
test_parser.add_argument('--checkpoint_glob', type=str, default=None,
                         help='Glob pattern for checkpoint directories to evaluate. '
                              'Defaults to ./checkpoint/<tag> when --tag/config tag is set.')
test_parser.add_argument('--ksteps', type=int, default=20,
                         help='Number of samples for stochastic evaluation')
test_parser.add_argument('--batch_size', type=int, default=1024,
                         help='batch size for test DataLoader (highD only; pedestrian uses 1)')
test_parser.add_argument('--num_workers', type=int, default=64,
                         help='number of DataLoader worker processes')
test_args = load_config_and_parse(test_parser)

if test_args.checkpoint_glob is None:
    tag = getattr(test_args, 'tag', None)
    test_args.checkpoint_glob = f'./checkpoint/{tag}' if tag else './checkpoint/*'

paths  = [test_args.checkpoint_glob]
KSTEPS = test_args.ksteps


for feta in range(len(paths)):
    ade_ls = []
    fde_ls = []
    path = paths[feta]
    exps = glob.glob(path)

    for exp_path in exps:
        print("Evaluating model:", exp_path)

        model_path = exp_path + '/val_best.pth'
        args_path  = exp_path + '/args.pkl'
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)

        #Data prep
        obs_seq_len  = args.obs_seq_len
        pred_seq_len = args.pred_seq_len

        if getattr(args, 'highd', False):
            from pathlib import Path
            feature_mode = getattr(args, 'feature_mode', 'baseline')
            input_feat   = get_input_size(feature_mode)
            splits_dir   = Path(getattr(args, 'splits_dir', 'data/highD/splits'))
            dset_test = HighDGraphDataset(
                    args.mmap_dir,
                    feature_mode=feature_mode,
                    pred_len=pred_seq_len,
                    norm_lap_matr=True,
                    indices_file=splits_dir / 'test_indices.npy')
        else:
            input_feat = getattr(args, 'input_size', 2)
            data_set = './datasets/' + args.dataset + '/'
            dset_test = TrajectoryDataset(
                    data_set + 'test/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1, norm_lap_matr=True)

        loader_batch_size = test_args.batch_size if getattr(args, 'highd', False) else 1
        loader_test = DataLoader(
                dset_test,
                batch_size=loader_batch_size,
                shuffle=False,
                num_workers=test_args.num_workers if getattr(args, 'highd', False) else 1,
                pin_memory=True)

        #Defining the model
        model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                input_feat=input_feat,
                output_feat=args.output_size, seq_len=args.obs_seq_len,
                kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))

        ade_  = 999999
        fde_  = 999999
        rmse_ = 999999
        print("Testing ....")
        ad, fd, rm, raw_data_dic_ = test()
        ade_  = min(ade_,  ad)
        fde_  = min(fde_,  fd)
        rmse_ = min(rmse_, rm)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print(f"\nADE: {ade_:.4f}  FDE: {fde_:.4f}  RMSE: {rmse_:.4f}")
