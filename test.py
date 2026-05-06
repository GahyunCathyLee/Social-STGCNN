import os
import math
import sys
import time
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
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Scenario label helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_scenario_labels(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing}")
        return None
    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label")
        return None
    lut = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }
    print(f"[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


def build_sample_label_list(mmap_dir, indices, labels_lut):
    mmap_dir = Path(mmap_dir)
    meta_rec   = np.load(mmap_dir / "meta_recordingId.npy", mmap_mode='r')
    meta_track = np.load(mmap_dir / "meta_trackId.npy",     mmap_mode='r')
    meta_frame = np.load(mmap_dir / "meta_frame.npy",       mmap_mode='r')
    sample_labels = []
    for idx in indices:
        key = (int(meta_rec[idx]), int(meta_track[idx]), int(meta_frame[idx]))
        sample_labels.append(labels_lut.get(key))
    return sample_labels


def _sep(widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * w for w in widths) + right


def print_scenario_results(stats, label_type):
    if not stats:
        return
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))
    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2
    c_n = 9
    c_m = 11
    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
          f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|")
    print(_sep(ws))

    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
              f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|")
    print(_sep(ws))

    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    total_n = sum(v[3] for v in stats.values())
    N = max(1, total_n)
    print(f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
          f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|")
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ──────────────────────────────────────────────────────────────────────────────

def print_device_info(device):
    """Print device info that affects inference latency."""
    print("\n====== Device Info ======")
    print(f"  PyTorch version : {torch.__version__}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        free_gb = total_gb - alloc_gb
        print(f"  Device          : {props.name}  (index={idx})")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"  SM count        : {props.multi_processor_count}")
        print(f"  VRAM            : {total_gb:.1f} GB total  /  {free_gb:.1f} GB free")
        print(f"  cuDNN benchmark : {torch.backends.cudnn.benchmark}")
    else:
        import platform
        cpu_name = platform.processor() or platform.machine() or "unknown"
        print(f"  Device          : CPU  ({cpu_name})")


@torch.no_grad()
def measure_latency(fn, device, warmup=1000, iters=10000):
    """
    Measure single-inference latency (avg / min / max) in milliseconds.

    CUDA: torch.cuda.Event based timing
    CPU : time.perf_counter based timing
    """
    print(f"  GPU warm-up  : {warmup:,} iters ...", end=" ", flush=True)
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")

    times_ms = []
    print(f"  Measurement  : {iters:,} iters ...", end=" ", flush=True)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    print("done")

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def print_latency(lat, batch_size, warmup, iters):
    """Print latency avg / min / max table."""
    c = 15
    ws = [c, c, c]

    print()
    print(f"  Batch size : {batch_size}   Warmup : {warmup:,}   Measurement : {iters:,}")
    print()
    print(_sep(ws))
    print(f"|{'Avg (ms)':^{c}}|{'Min (ms)':^{c}}|{'Max (ms)':^{c}}|")
    print(_sep(ws))
    print(f"|{lat['avg_ms']:^{c}.2f}|{lat['min_ms']:^{c}.2f}|{lat['max_ms']:^{c}.2f}|")
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Test function
# ──────────────────────────────────────────────────────────────────────────────

def test(KSTEPS=20, sample_labels=None):
    global loader_test, model, args
    model.eval()
    ade_bigls  = []
    fde_bigls  = []
    rmse_bigls = []
    raw_data_dict = {}
    step = 0

    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    sample_cursor = 0

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

                batch_ade  = ade_all.min(axis=0)   # (B,)
                batch_fde  = fde_all.min(axis=0)   # (B,)

                ade_bigls.extend(batch_ade.tolist())
                fde_bigls.extend(batch_fde.tolist())
                rmse_bigls.extend(rmse_per.tolist())

                # Per-scenario accumulation
                if sample_labels is not None:
                    for i in range(B):
                        if sample_cursor + i >= len(sample_labels):
                            break
                        lab = sample_labels[sample_cursor + i]
                        if lab is None:
                            continue
                        ev = lab.get("event_label") or "unknown"
                        st = lab.get("state_label") or "unknown"
                        for acc, lbl in ((ev_stats, ev), (st_stats, st)):
                            acc[lbl][0] += float(batch_ade[i])
                            acc[lbl][1] += float(batch_fde[i])
                            acc[lbl][2] += float(rmse_per[i])
                            acc[lbl][3] += 1

                sample_cursor += B

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
    return ade_, fde_, rmse_, raw_data_dict, ev_stats, st_stats


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
test_parser.add_argument('--scenario_labels', type=str, default=None,
                         help='Path to scenario_labels.csv for per-scenario breakdown')
test_parser.add_argument('--measure_time', action='store_true',
                         help='Measure inference latency (1,000 warmup + 10,000 iters)')
test_args = load_config_and_parse(test_parser)

LATENCY_WARMUP = 1_000
LATENCY_ITERS  = 10_000

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
        model.eval()

        if test_args.measure_time:
            device = next(model.parameters()).device
            print_device_info(device)

            # Single sample (batch_size=1) for accurate per-inference latency.
            sample = dset_test[0]
            V_obs_l = sample[6].unsqueeze(0).to(device)
            A_obs_l = sample[7].unsqueeze(0).to(device)
            V_obs_tmp_l = V_obs_l.permute(0, 3, 1, 2)

            @torch.no_grad()
            def _infer():
                V_pred_l, _ = model(V_obs_tmp_l, A_obs_l)
                V_pred_l.permute(0, 2, 3, 1)

            print(f"\n====== Inference Latency ======")
            lat = measure_latency(
                _infer,
                device,
                warmup=LATENCY_WARMUP,
                iters=LATENCY_ITERS,
            )
            print_latency(
                lat,
                batch_size=1,
                warmup=LATENCY_WARMUP,
                iters=LATENCY_ITERS,
            )
            continue

        # Build scenario labels
        sample_labels = None
        if getattr(args, 'highd', False):
            scenario_labels_path = test_args.scenario_labels
            if scenario_labels_path is None:
                candidate = Path(args.mmap_dir) / 'scenario_labels.csv'
                if candidate.exists():
                    scenario_labels_path = str(candidate)
            if scenario_labels_path:
                labels_lut = load_scenario_labels(scenario_labels_path)
                if labels_lut is not None:
                    sample_labels = build_sample_label_list(
                        args.mmap_dir, dset_test.indices, labels_lut)

        ade_  = 999999
        fde_  = 999999
        rmse_ = 999999
        print("Testing ....")
        ad, fd, rm, raw_data_dic_, ev_stats, st_stats = test(
            KSTEPS=KSTEPS, sample_labels=sample_labels)
        ade_  = min(ade_,  ad)
        fde_  = min(fde_,  fd)
        rmse_ = min(rmse_, rm)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print(f"\nADE: {ade_:.4f}  FDE: {fde_:.4f}  RMSE: {rmse_:.4f}")

        if ev_stats:
            print_scenario_results(ev_stats, label_type="Event")
        if st_stats:
            print_scenario_results(st_stats, label_type="State")
