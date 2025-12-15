import os
import json
import argparse
import numpy as np
import time
import scipy.io
import h5py
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import logging
import network

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", default="./data/star10_kh9_10_10_n48_2000_noise0", type=str)
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument("--train_cfg_path", default=None, type=str)

    parser.add_argument("--retrain", default=None, type=str)
    parser.add_argument("--resume_ckpt", default=None, type=str)

    parser.add_argument("--ndata_train", default=None, type=int)

    parser.add_argument("--k_start", default=0, type=int)
    parser.add_argument("--k_end", default=None, type=int)

    parser.add_argument("--epochs", default=400, type=int)

    parser.add_argument("--chunk_files", default=2, type=int)
    parser.add_argument("--shuffle_files", action="store_true")

    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--save_every_epochs", default=25, type=int)

    args = parser.parse_args()

    if args.retrain and args.train_cfg_path is None:
        retrain_str = args.retrain
        sep = "/" if "/" in retrain_str else "\\"
        old_model_name = retrain_str.split(sep)[0]
        args.train_cfg_path = os.path.join(args.dirname, old_model_name, "train_config.json")

    if args.train_cfg_path is None:
        dirname = os.path.basename(args.dirname)
        ncstr = dirname.split('_')[0]
        if ncstr.startswith("star"):
            nc = int(ncstr[4:])
            args.train_cfg_path = f"./configs/train_nc{nc}.json"
        else:
            raise ValueError("Cannot infer train_cfg_path; please pass --train_cfg_path")

    with open(args.train_cfg_path, "r") as f:
        train_cfg = json.load(f)

    if train_cfg.get("network_type", "convnet") != "convnet":
        raise ValueError("This train.py rewrite is convnet-only. Set network_type='convnet'.")

    return args, train_cfg


def list_train_files(train_dir: str):
    files = []
    for fn in os.listdir(train_dir):
        if fn.startswith("train_data_") and fn.endswith(".mat"):
            files.append(os.path.join(train_dir, fn))
    files.sort()
    return files


def _decode_h5py_complex(ds):
    """
    ds is a numpy array from h5py reading that may be:
      - complex already (rare)
      - compound dtype with fields ('real','imag') (common for MATLAB v7.3 complex)
    returns complex ndarray
    """
    arr = ds
    if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
        fields = arr.dtype.fields
        if "real" in fields and "imag" in fields:
            return arr["real"] + 1j * arr["imag"]
    return arr

def read_train_mat_v73_k_slice(path: str, k: int, nk_expected: int, n_per_file_expected: int):
    """
    Robust v7.3 reader that extracts ONLY the k-slice, regardless of HDF5 dimension order.

    Returns:
      coefs:   (N, nc)
      uscat_k: (N, 1, H, W) complex
      timings: dict
    """
    t0 = time.time()
    with h5py.File(path, "r") as f:
        t_open = time.time()

        coefs = f["coefs"][()]
        t_coefs = time.time()

        # Fix coefs to (N, nc)
        # Could be (N, nc) or (nc, N)
        if coefs.ndim != 2:
            raise ValueError(f"Unexpected coefs ndim={coefs.ndim} in {path}")
        if coefs.shape[0] == n_per_file_expected:
            N = coefs.shape[0]
            # already (N, nc)
        elif coefs.shape[1] == n_per_file_expected:
            coefs = coefs.T
            N = coefs.shape[0]
        else:
            raise ValueError(f"Cannot determine N from coefs shape {coefs.shape} in {path}")

        uscat_ds = f["uscat"]
        shp = uscat_ds.shape

        # Find nk axis (should be unique, e.g. 30)
        nk_axes = [i for i, d in enumerate(shp) if d == nk_expected]
        if len(nk_axes) != 1:
            raise ValueError(f"Expected exactly one nk axis size={nk_expected}, got axes={nk_axes} for uscat shape={shp} in {path}")
        ax_k = nk_axes[0]

        # Find N axis (per-file sample axis, e.g. 100)
        n_axes = [i for i, d in enumerate(shp) if d == N]
        if len(n_axes) < 1:
            raise ValueError(f"Could not find N axis size={N} in uscat shape={shp} in {path}")

        # If multiple axes equal N (rare), prefer the last one (common MATLAB layout has N last)
        ax_n = n_axes[-1]

        # Slice only the k plane
        slc = [slice(None)] * len(shp)
        slc[ax_k] = k
        uscat_k = uscat_ds[tuple(slc)]  # now 3D
        t_uscat = time.time()

    # Decode MATLAB complex (compound real/imag)
    uscat_k = _decode_h5py_complex(uscat_k)

    # uscat_k is now 3D, but in some order. Move N axis to the front.
    # Note: after slicing, the axis indices shift if ax_k < ax_n.
    # Compute new index of N axis after removing ax_k.
    ax_n_after = ax_n - 1 if ax_k < ax_n else ax_n
    uscat_k = np.moveaxis(uscat_k, ax_n_after, 0)  # (N, ?, ?)

    # Finally add the channel dimension: (N,1,H,W)
    if uscat_k.ndim != 3:
        raise ValueError(f"Expected uscat_k to be 3D after moveaxis, got shape {uscat_k.shape} in {path}")
    uscat_k = uscat_k[:, None, :, :]

    timings = {
        "t_open": t_open - t0,
        "t_coefs": t_coefs - t_open,
        "t_uscat": t_uscat - t_coefs,
        "uscat_shape": tuple(shp),
        "ax_k": ax_k,
        "ax_n": ax_n,
        "used_fallback_full_read": False,
    }
    return coefs, uscat_k, timings

def save_checkpoint(path, model, optimizer, scheduler, progress: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "progress": progress,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, map_location):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("progress", {})


def main():
    start_time = time.time()
    args, train_cfg = parse_args()

    if train_cfg["data_type"] == "float32":
        data_type = torch.float32
        np_dtype = np.float32
    elif train_cfg["data_type"] == "float64":
        data_type = torch.float64
        np_dtype = np.float64
    else:
        raise ValueError("Unsupported data_type in train_cfg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("train data from %s", args.dirname)
    logger.info("model name %s", args.model_name)

    # -------------------------
    # Load validation (v7.3)
    # -------------------------
    valid_path = os.path.join(args.dirname, "valid_data.mat")
    with h5py.File(valid_path, "r") as f:
        coefs_val = f["coefs_val"][()]
        coefs_val = torch.from_numpy(coefs_val).permute(1, 0)

        uscat_val = f["uscat_val"][()]
        uscat_val = uscat_val["real"] + 1j * uscat_val["imag"]
        uscat_val = np.transpose(uscat_val, (3, 2, 0, 1))

        cfg_str_data = f["cfg_str"][()]
        cfg_str = "".join(chr(x) for x in np.nditer(cfg_str_data))
        data_cfg = json.loads(cfg_str)

    norm_coef = np.linalg.norm(coefs_val.numpy(), axis=1)

    if train_cfg.get("n_dir_train", 0) > 0:
        uscat_val = uscat_val[:, :, 0:train_cfg["n_dir_train"], :]
    if train_cfg.get("n_tgt_train", 0) > 0:
        uscat_val = uscat_val[:, :, :, 0:train_cfg["n_tgt_train"]]

    tgt_valid = uscat_val.real
    mean = float(np.mean(tgt_valid))
    std = float(np.std(tgt_valid))
    logger.info("convnet mean %.8e std %.8e", mean, std)
    tgt_valid = (tgt_valid - mean) / std

    nk = int(tgt_valid.shape[1])
    k_start = int(args.k_start)
    k_end = int(args.k_end) if args.k_end is not None else (nk - 1)
    if not (0 <= k_start <= k_end < nk):
        raise ValueError(f"Invalid k range: k_start={k_start}, k_end={k_end}, nk={nk}")

    # -------------------------
    # Setup output dirs
    # -------------------------
    model_dir = os.path.join(args.dirname, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "inverse"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "figs"), exist_ok=True)

    writer = SummaryWriter(model_dir)

    with open(os.path.join(model_dir, "mean_std.txt"), "w") as f:
        f.write(f"{mean}\n{std}\n")
    with open(os.path.join(model_dir, "data_config.json"), "w") as f:
        json.dump(data_cfg, f)
    with open(os.path.join(model_dir, "train_config.json"), "w") as f:
        json.dump(train_cfg, f)

    # -------------------------
    # Model / optimizer
    # -------------------------
    model = network.ConvNet(data_cfg, train_cfg).to(device).type(data_type)

    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    else:
        raise ValueError("Unsupported optimizer")

    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    loss_fn = nn.MSELoss()

    tgt_valid_t = torch.tensor(tgt_valid, dtype=data_type)
    coefs_val_t = torch.tensor(coefs_val.numpy(), dtype=data_type)

    # -------------------------
    # Train file list
    # -------------------------
    train_dir = os.path.join(args.dirname, "train_data")
    train_files = list_train_files(train_dir)
    if len(train_files) == 0:
        raise RuntimeError(f"No train_data_*.mat files found in {train_dir}")

    ndata_per_mat = int(data_cfg["ndata_per_mat"])
    ndata_avail = len(train_files) * ndata_per_mat

    ndata = train_cfg.get("ndata_train", 0)
    if args.ndata_train is not None:
        ndata = args.ndata_train
    if ndata is None or ndata == 0:
        ndata = ndata_avail
    else:
        ndata = min(int(ndata), int(ndata_avail))

    n_files_needed = int(np.ceil(ndata / ndata_per_mat))
    train_files = train_files[:n_files_needed]
    logger.info("Using %d train files (~%d samples)", len(train_files), len(train_files) * ndata_per_mat)

    # -------------------------
    # Resume / retrain
    # -------------------------
    progress = {
        "k": k_start,
        "epoch_in_k": 0,
        "global_step": 0,
        "best_rel": 1.0,
        "best_abs": 1e9,
    }

    if args.resume_ckpt:
        logger.info("Resuming from %s", args.resume_ckpt)
        loaded = load_checkpoint(args.resume_ckpt, model, optimizer, scheduler, map_location=device)
        progress.update({kk: loaded[kk] for kk in loaded.keys() if kk in progress})
        logger.info("Loaded progress: %s", str(progress))
    elif args.retrain:
        logger.info("Warm-starting weights from %s", args.retrain)
        retrain_path = args.retrain
        if not os.path.isabs(retrain_path):
            retrain_path = os.path.join(args.dirname, args.retrain)
        model.load_state_dict(torch.load(retrain_path, map_location=device))

    def iter_chunks(files, chunk_files):
        for i in range(0, len(files), chunk_files):
            yield i // chunk_files, files[i:i + chunk_files]

    # -------------------------
    # Training: stage-by-k
    # -------------------------
    logger.info("Curriculum k: %d..%d | epochs per k-stage: %d", k_start, k_end, args.epochs)
    logger.info("Chunking: %d files/chunk", args.chunk_files)

    for k in range(progress["k"], k_end + 1):
        start_epoch = progress["epoch_in_k"] if k == progress["k"] else 0

        for e in range(start_epoch, args.epochs):
            stage_files = list(train_files)
            if args.shuffle_files:
                rng = np.random.default_rng(seed=12345 + 1000 * k + e)
                rng.shuffle(stage_files)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for chunk_idx, file_list in iter_chunks(stage_files, args.chunk_files):
                t_chunk0 = time.time()

                xs = []
                ys = []
                t_open_sum = t_coefs_sum = t_uscat_sum = 0.0
                used_fallback = False

                # Read only needed k-slice from each file
                for fp in file_list:
                    t_file0 = time.time()
                    coefs_i, uscat_k_i, tinfo = read_train_mat_v73_k_slice(
                        fp,
                        k,
                        nk_expected=nk,                 # from tgt_valid.shape[1]
                        n_per_file_expected=ndata_per_mat  # from data_cfg["ndata_per_mat"]
                    )

                    t_file1 = time.time()

                    t_open_sum += tinfo["t_open"]
                    t_coefs_sum += tinfo["t_coefs"]
                    t_uscat_sum += tinfo["t_uscat"]
                    used_fallback = used_fallback or tinfo["used_fallback_full_read"]

                    # Apply partial dir/tgt if desired
                    if train_cfg.get("n_dir_train", 0) > 0:
                        uscat_k_i = uscat_k_i[:, :, 0:train_cfg["n_dir_train"], :]
                    if train_cfg.get("n_tgt_train", 0) > 0:
                        uscat_k_i = uscat_k_i[:, :, :, 0:train_cfg["n_tgt_train"]]

                    # Convnet input: real part, normalize
                    x_i = uscat_k_i.real  # (N,1,H,W)
                    x_i = ((x_i - mean) / std).astype(np_dtype)
                   #  logger.info("DEBUG x_i shape after slice / normalize %s", x_i.shape)
                    y_i = coefs_i.astype(np_dtype)

                    xs.append(x_i)
                    ys.append(y_i)
                    if chunk_idx < 1:
                      logger.info(
                          "k=%d e=%d chunk=%d loaded %s in %.2fs (open %.2fs, coefs %.2fs, uscat %.2fs)%s",
                          k, e, chunk_idx, os.path.basename(fp), (t_file1 - t_file0),
                          tinfo["t_open"], tinfo["t_coefs"], tinfo["t_uscat"],
                          " [FALLBACK FULL READ]" if tinfo["used_fallback_full_read"] else ""
                      )

                t_stack0 = time.time()
                x = np.vstack(xs)
                y = np.vstack(ys)
                t_stack1 = time.time()

                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x),
                    torch.from_numpy(y)
                )

                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=train_cfg["batch_size"],
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(args.num_workers > 0),
                )

                t_build = time.time()

                # Train on this chunk
                model.train()
                for xb, yb in loader:
                    xb = xb.to(device).type(data_type)
                    yb = yb.to(device).type(data_type)
                    # if chunk_idx == 0 and e == 0:
                       # logger.info("DEBUG xb shape: %s", tuple(xb.shape))
                        # logger.info("DEBUG fc1 expects in_features=%d", model.fc1.in_features)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss_sum += float(loss.item())
                    epoch_loss_count += 1
                    progress["global_step"] += 1

                t_train = time.time()
                if chunk_idx == 0:
                     logger.info(
                        "k=%d e=%d chunk=%d loaded %s in %.2fs (open %.2fs, coefs %.2fs, uscat %.2fs) uscat_shape=%s ax_k=%d ax_n=%d",
                        k, e, chunk_idx, os.path.basename(fp), (t_file1 - t_file0),
                        tinfo["t_open"], tinfo["t_coefs"], tinfo["t_uscat"],
                        tinfo["uscat_shape"], tinfo["ax_k"], tinfo["ax_n"]
                    )

                del dataset, loader, x, y, xs, ys, uscat_k_i, coefs_i
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            scheduler.step()

            if e % train_cfg["valid_freq"] == 0:
                model.eval()
                with torch.no_grad():
                    coef_pred = model(tgt_valid_t[:, k:k + 1, :, :].to(device))
                    loss_train = epoch_loss_sum / max(epoch_loss_count, 1)
                    loss_val = loss_fn(coef_pred, coefs_val_t.to(device)).item()

                    diff = torch.norm(coef_pred.cpu() - coefs_val_t, dim=1).cpu().numpy()
                    err_rel = float(np.mean(diff / norm_coef))
                    err_abs = float(np.mean(diff))

                    progress["best_rel"] = min(progress["best_rel"], err_rel)
                    progress["best_abs"] = min(progress["best_abs"], err_abs)

                    logger.info(
                        "k=%2d e=%4d train_loss=%.6f val_loss=%.6f rel=%.4f abs=%.4f time=%.1fs",
                        k, e, loss_train, loss_val, err_rel, err_abs, time.time() - start_time
                    )

                    writer.add_scalar("loss_train", loss_train, progress["global_step"])
                    writer.add_scalar("loss_val", loss_val, progress["global_step"])

            if args.save_every_epochs > 0 and (e % args.save_every_epochs == 0):
                progress["k"] = k
                progress["epoch_in_k"] = e+1
                save_checkpoint(
                    os.path.join(model_dir, "checkpoints", "ckpt_latest.pt"),
                    model, optimizer, scheduler, progress
                )

        torch.save(model.state_dict(), os.path.join(model_dir, f"model_k{k}.pt"))

        progress["k"] = k + 1
        progress["epoch_in_k"] = 0
        save_checkpoint(
            os.path.join(model_dir, "checkpoints", "ckpt_latest.pt"),
            model, optimizer, scheduler, progress
        )

    logger.info("DONE. Best rel %.4f best abs %.4f", progress["best_rel"], progress["best_abs"])

    last_k = k_end
    model.eval()
    with torch.no_grad():
        coef_pred = model(tgt_valid_t[:, last_k:last_k + 1, :, :].to(device))

    scipy.io.savemat(
        os.path.join(args.dirname, f"valid_predby_{args.model_name}.mat"),
        {
            "coef_val": coefs_val_t.cpu().numpy().astype("float64"),
            "coef_pred": coef_pred.detach().cpu().numpy().astype("float64"),
            "cfg_str": data_cfg
        }
    )

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    writer.close()


if __name__ == "__main__":
    main()

