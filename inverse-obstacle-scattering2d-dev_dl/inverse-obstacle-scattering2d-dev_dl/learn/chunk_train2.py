import os
import json
import argparse
import numpy as np
import time
import h5py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import logging
import network

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    parser.add_argument(
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs to train each k in this invocation.",
    )
    parser.add_argument("--chunk_files", default=16, type=int) # Acts as batch size for file loader
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)  # CRITICAL: Controls parallelism
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

    return args, train_cfg

def list_train_files(train_dir: str):
    files = []
    for fn in os.listdir(train_dir):
        if fn.startswith("train_data_") and fn.endswith(".mat"):
            files.append(os.path.join(train_dir, fn))
    files.sort()
    return files

def _decode_h5py_complex(arr):
    if isinstance(arr, np.ndarray) and arr.dtype.fields is not None:
        fields = arr.dtype.fields
        if "real" in fields and "imag" in fields:
            return arr["real"] + 1j * arr["imag"]
    return arr

# NOTE: This function is now used by workers in separate processes
def read_single_file(path, k, nk_expected, n_per_file_expected, n_dir, n_tgt, mean, std, np_dtype):
    try:
        with h5py.File(path, "r") as f:
            coefs = f["coefs"][()]
            
            # Robust shape check for coefs
            if coefs.ndim != 2:
                # Fallback or error
                pass 
            
            if coefs.shape[0] == n_per_file_expected:
                N = coefs.shape[0]
            elif coefs.shape[1] == n_per_file_expected:
                coefs = coefs.T
                N = coefs.shape[0]
            else:
                # Fallback for N detection
                N = 100 # Assuming standard if check fails, or raise error
            
            uscat_ds = f["uscat"]
            shp = uscat_ds.shape
            
            # Logic to find axes (simplified for robustness in worker)
            # Assuming standard layout if complex logic fails, but keeping original logic:
            nk_axes = [i for i, d in enumerate(shp) if d == nk_expected]
            ax_k = nk_axes[0]
            
            # Reading slice
            slc = [slice(None)] * len(shp)
            slc[ax_k] = k
            uscat_k = uscat_ds[tuple(slc)]

        uscat_k = _decode_h5py_complex(uscat_k)
        
        # Reshape logic (simplified: we know we want (N, 1, H, W))
        # If uscat is (N, dir, tgt), we assume H, W comes from dir/tgt or similar.
        # Original script logic for moveaxis:
        # We just need to ensure N is at dim 0.
        # If N was at the end, it might still be at the end.
        if uscat_k.shape[-1] == N:
            uscat_k = np.moveaxis(uscat_k, -1, 0)
        elif uscat_k.shape[0] != N:
             # Try to find N axis
             pass

        # Add channel dim if missing
        if uscat_k.ndim == 3: 
             uscat_k = uscat_k[:, None, :, :]

        # Slicing
        if n_dir > 0:
            uscat_k = uscat_k[:, :, 0:n_dir, :]
        if n_tgt > 0:
            uscat_k = uscat_k[:, :, :, 0:n_tgt]

        # Normalize
        x = uscat_k.real
        x = ((x - mean) / std).astype(np_dtype)
        y = coefs.astype(np_dtype)
        
        return x, y
    except Exception as e:
        # If a worker fails, print why (since workers silence stdout sometimes)
        print(f"Error reading {path}: {e}")
        raise e

# --- DATASET FOR MULTIPROCESSING ---
class HDF5FileDataset(Dataset):
    def __init__(self, file_paths, k, nk, ndata_per_mat, n_dir, n_tgt, mean, std, np_dtype):
        self.file_paths = file_paths
        self.k = k
        self.nk = nk
        self.ndata_per_mat = ndata_per_mat
        self.n_dir = n_dir
        self.n_tgt = n_tgt
        self.mean = mean
        self.std = std
        self.np_dtype = np_dtype

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # This executes in a separate process!
        path = self.file_paths[idx]
        x, y = read_single_file(
            path, self.k, self.nk, self.ndata_per_mat, 
            self.n_dir, self.n_tgt, self.mean, self.std, self.np_dtype
        )
        return x, y

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
    print(torch.cuda.get_device_name(0))
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
    use_cuda = (device.type == "cuda")

    # -------------------------
    # Validation Data Load
    # -------------------------
    logger.info("train data from %s", args.dirname)
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
    tgt_valid = (tgt_valid - mean) / std  # (nvalid, nk, H, W)

    nk = int(tgt_valid.shape[1])
    k_start = int(args.k_start)
    k_end = int(args.k_end) if args.k_end is not None else (nk - 1)

    # -------------------------
    # Setup Output
    # -------------------------
    model_dir = os.path.join(args.dirname, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(model_dir)

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
    # Model Setup
    # -------------------------
    model = network.ConvNet(data_cfg, train_cfg).to(device).type(data_type)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"])
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    tgt_valid_t = torch.tensor(tgt_valid, dtype=data_type)
    coefs_val_t = torch.tensor(coefs_val.numpy(), dtype=data_type)

    # -------------------------
    # File List
    # -------------------------
    train_dir = os.path.join(args.dirname, "train_data")
    train_files = list_train_files(train_dir)
    ndata_per_mat = int(data_cfg["ndata_per_mat"])

    # -------------------------
    # Checkpoint Logic
    # -------------------------
    progress = {"k": k_start, "epoch_in_k": 0, "global_step": 0, "best_rel": 1.0, "best_abs": 1e9}

    if args.resume_ckpt:
        logger.info("Resuming from %s", args.resume_ckpt)
        loaded = load_checkpoint(args.resume_ckpt, model, optimizer, scheduler, map_location=device)
        progress.update({kk: loaded[kk] for kk in loaded.keys() if kk in progress})
    elif args.retrain:
        logger.info("Warm-starting weights from %s", args.retrain)
        retrain_path = args.retrain
        if not os.path.isabs(retrain_path):
            retrain_path = os.path.join(args.dirname, args.retrain)
        model.load_state_dict(torch.load(retrain_path, map_location=device))

    # -------------------------
    # Training Loop
    # -------------------------
    bs = int(train_cfg["batch_size"])
    valid_freq = int(train_cfg["valid_freq"])
    
    # We use batch_size in DataLoader to serve as "chunk_size" (files per chunk)
    files_per_chunk = args.chunk_files 

    for k in range(progress["k"], k_end + 1):
        start_epoch = progress["epoch_in_k"] if k == progress["k"] else 0
        
        # Create Dataset for this k
        dataset = HDF5FileDataset(
            train_files, k, nk, ndata_per_mat, 
            train_cfg.get("n_dir_train", 0), train_cfg.get("n_tgt_train", 0), 
            mean, std, np_dtype
        )

        # Create Loader with MULTIPROCESSING (num_workers > 0)
        # This spawns processes that read HDF5 in parallel without locking each other
        loader = DataLoader(
            dataset,
            batch_size=files_per_chunk,
            shuffle=args.shuffle_files,
            num_workers=args.num_workers, # 8 workers = 8x speedup on I/O
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False
        )
        # The shell launchers select the desired regime for this k range
        # (for example 60, 120, or 240 epochs).  Do not override that
        # command-line value with a separate hard-coded schedule here.
        target_epochs = args.epochs
        logger.info(f"Starting stage k={k} with {target_epochs} epochs")
        for e in range(start_epoch, target_epochs):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            
            # The loader yields batches of files: 
            # x_files shape: (files_per_chunk, N, 1, H, W)
            for chunk_idx, (x_files, y_files) in enumerate(loader):
                if chunk_idx == 0 and e == start_epoch:
                    logger.info("First chunk loaded! Shape: %s", str(x_files.shape))

                # Flatten files into one big batch
                # (B_files, N, C, H, W) -> (B_files*N, C, H, W)
                Bf, N, C, H, W = x_files.shape
                x_cpu = x_files.view(-1, C, H, W)
                y_cpu = y_files.view(-1, y_files.shape[-1])

                # Shuffle
                perm = torch.randperm(x_cpu.shape[0])
                x_cpu = x_cpu[perm]
                y_cpu = y_cpu[perm]

                model.train()
                n_chunk = x_cpu.shape[0]

                # Inner Loop
                for i0 in range(0, n_chunk, bs):
                    xb = x_cpu[i0:i0 + bs].to(device, non_blocking=True).type(data_type)
                    yb = y_cpu[i0:i0 + bs].to(device, non_blocking=True).type(data_type)

                    if use_cuda: xb = xb.to(memory_format=torch.channels_last)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=use_cuda):
                        pred = model(xb)
                        loss = loss_fn(pred, yb)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss_sum += float(loss.item())
                    epoch_loss_count += 1
                    progress["global_step"] += 1
                
                # Cleanup to save RAM
                del x_cpu, y_cpu, xb, yb

            scheduler.step()

            # Validation
            if (e % valid_freq) == 0:
                model.eval()
                with torch.no_grad():
                    xb_val = tgt_valid_t[:, k:k + 1, :, :].to(device).type(data_type)
                    if use_cuda: xb_val = xb_val.to(memory_format=torch.channels_last)
                    coef_pred = model(xb_val)
                    
                    loss_train = epoch_loss_sum / max(epoch_loss_count, 1)
                    loss_val = loss_fn(coef_pred, coefs_val_t.to(device).type(data_type)).item()
                    
                    diff = torch.norm(coef_pred.detach().cpu() - coefs_val_t, dim=1).cpu().numpy()
                    err_rel, err_abs = float(np.mean(diff / norm_coef)), float(np.mean(diff))

                    progress["best_rel"] = min(progress["best_rel"], err_rel)
                    progress["best_abs"] = min(progress["best_abs"], err_abs)
                    
                    logger.info("k=%2d e=%4d train_loss=%.6f val_loss=%.6f rel=%.4f abs=%.4f time=%.1fs",
                        k, e, loss_train, loss_val, err_rel, err_abs, time.time() - start_time)
                    writer.add_scalar("loss_train", loss_train, progress["global_step"])

            # Save Checkpoint
            if args.save_every_epochs > 0 and (e % args.save_every_epochs == 0):
                progress["k"] = k
                progress["epoch_in_k"] = e + 1
                save_checkpoint(os.path.join(model_dir, "checkpoints", "ckpt_latest.pt"), model, optimizer, scheduler, progress)

        # End of K
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_k{k}.pt"))
        progress["k"] = k + 1
        progress["epoch_in_k"] = 0
        save_checkpoint(os.path.join(model_dir, "checkpoints", "ckpt_latest.pt"), model, optimizer, scheduler, progress)

    logger.info("DONE")
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    writer.close()

if __name__ == "__main__":
    # Standard boilerplate for safe multiprocessing on Linux
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
