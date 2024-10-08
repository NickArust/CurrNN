import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import scipy.fftpack as sfft
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import logging
from multiprocessing import Pool
import network
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
torch.backends.cudnn.benchmark = True

#-------------------Nick's Changes-------------

# def checkpoint(model, optimizer, filename):
#     torch.save({
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             #'loss': LOSS,
#             }, filename)

# def resume(model,optimizer, filename): 
#     checkpoint = torch.load(filename)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#----------End of Changes----------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", default="./data/star5_kh10_n48_600", type=str)
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument("--train_cfg_path", default=None, type=str)
    parser.add_argument("--retrain", default=None, type=str) #format: test/model_100.pt
    parser.add_argument("--ndata_train", default=None, type=int)
    parser.add_argument("--cfg_by_nc", action='store_true') #default False
    args = parser.parse_args()
    if args.retrain:
        old_model_name = args.retrain[:args.retrain.find('/' or "\\")]
        args.train_cfg_path = os.path.join(args.dirname, old_model_name, "train_config.json")
    elif args.train_cfg_path is None:
        dirname = os.path.basename(args.dirname)
        ncstr = dirname.split('_')[0]
        if ncstr.startswith("star"):
            try:
                nc = int(ncstr[4:])
            except ValueError:
                logger.error("cannot get the default training config path from dirname.")
                raise
        args.train_cfg_path = "./configs/train_nc{}.json".format(nc)
    print()
    f = open(args.train_cfg_path)
    train_cfg = json.load(f)
    f.close()
    return args, train_cfg

def read_data(data_dir):
    data = scipy.io.loadmat(data_dir)
    return data["coefs"], data["uscat"]

def main():
    start_time = time.time()
    args, train_cfg = parse_args()
    if train_cfg["data_type"] == "float32": data_type = torch.float32
    elif train_cfg["data_type"] == "float64": data_type = torch.float64
    logger.info("train data from {}".format(args.dirname))
    logger.info("model name {}".format(args.model_name))
    fname = os.path.join(args.dirname, "valid_data.mat")
    network_type = train_cfg["network_type"]
    valid_data = scipy.io.loadmat(fname)
    data_cfg = json.loads(valid_data["cfg_str"][0])
    coef_val = valid_data["coefs_val"]
    uscat_val = valid_data["uscat_val"]
    norm_coef = np.linalg.norm(coef_val, axis=1)
    if train_cfg["n_dir_train"] > 0:
        uscat_val = uscat_val[:,:,0:train_cfg["n_dir_train"],:]
    if train_cfg["n_tgt_train"] > 0:
        uscat_val = uscat_val[:,:,:,0:train_cfg["n_tgt_train"]]
        
    # change the train config for the error curve
    if args.cfg_by_nc:
        nc = data_cfg["nc"]
        logger.info("use train config with nc={:3} for the error curve".format(nc))
        if data_cfg["fc_max"]!=0.1 or data_cfg["n_tgt"]!=100 or data_cfg["n_dir"]!=100:
            logger.warning("data config does not match the setting for error curve")
        train_cfg["batch_size"] = 100
        train_cfg["epoch"] = 5000
        train_cfg["save_every_nepoch"] = 0
        train_cfg["milestones"] = [4900]
        train_cfg["out_channels"] = nc
        train_cfg["kernel_size"] = 9
        train_cfg["paddle"] = 4
        train_cfg["linear_dim"] = [50*nc, 10*nc]
    
    if network_type == 'convnet':
        tgt_valid = uscat_val.real
        mean = np.mean(tgt_valid)
        std = np.std(tgt_valid)
        logger.info("model convnet, mean %.8e, std %.8e", mean, std)
        tgt_valid = (tgt_valid-mean) / std
    elif network_type == 'complexnet':
        uscat_ft = sfft.fft2(uscat_val)
        uscat_ft_shift = sfft.fftshift(uscat_ft,axes=(1,2))
        data_real = uscat_ft_shift.real
        data_imag = uscat_ft_shift.imag
        mean_r = np.mean(data_real)
        mean_i = np.mean(data_imag)
        std_r = np.std(data_real)
        std_i = np.std(data_imag)
        logger.info("model complexnet, mean and std for real %.8e %.8e, imag %.8e %.8e", mean_r, std_r, mean_i, std_i)
        data_real = (data_real[:, None, :, :]-mean_r) / std_r
        data_imag = (data_imag[:, None, :, :]-mean_i) / std_i
        tgt_valid = np.concatenate((data_real, data_imag), axis=1)
        
    model_dir=os.path.join(args.dirname, args.model_name)
    writer = SummaryWriter(model_dir) # will create model_name folder
    total_data = data_cfg["ndata"]
    os.makedirs(os.path.join(model_dir,"inverse"), exist_ok=True)
    os.makedirs(os.path.join(model_dir,"figs"), exist_ok=True)
    f = open(os.path.join(model_dir, "mean_std.txt"), 'w')
    if network_type == 'convnet':
        f.writelines(f"{mean}\n{std}")
    elif network_type == 'complexnet':
        f.writelines(f"{mean_r}\n{std_r}\n{mean_i}\n{std_i}")
    f.close()
    g = open(os.path.join(model_dir, "data_config.json"), 'w')
    json.dump(data_cfg, g)
    g.close()
    h = open(os.path.join(model_dir, "train_config.json"), 'w')
    json.dump(train_cfg, h)
    h.close()
    
    # load training data
    data_dir = os.path.join(args.dirname, "train_data")
    ndata = train_cfg["ndata_train"]
    if args.ndata_train: ndata = args.ndata_train
    ndata_per_mat = data_cfg["ndata_per_mat"]
    ndata_avail = len(os.listdir(data_dir)) * ndata_per_mat
    if ndata == 0: 
        ndata = ndata_avail
    elif ndata > ndata_avail:
        logger.warning("ndata_train={:3} out numbers the data_set, will train with ndata={:3}".format(ndata, ndata_avail))
        ndata = ndata_avail
    nmat = int(np.ceil(ndata / ndata_per_mat))
    pool = Pool()
    temp_dir = os.path.join(data_dir, "train_data_")
    data_all = pool.map(read_data, [temp_dir+str((mat_id-1)*ndata_per_mat+1)+'-'+str(mat_id*ndata_per_mat)+'.mat'
                                    for mat_id in range(1,nmat+1)])
    coefs_all = (np.vstack([data[0] for data in data_all]))[:ndata,:]

    
    uscat_all = (np.vstack([data[1] for data in data_all]))[:ndata,:,:,:]
    if train_cfg["n_dir_train"] > 0:
        logger.info("training using partial data: num incident direction {:3}".format(train_cfg["n_dir_train"]))
        uscat_all = uscat_all[:,:,0:train_cfg["n_dir_train"],:]
    if train_cfg["n_tgt_train"] > 0:
        logger.info("training using partial data: num scattered direction {:3}".format(train_cfg["n_tgt_train"]))
        uscat_all = uscat_all[:,:,:,0:train_cfg["n_tgt_train"]]
    del data_all
    if network_type == 'convnet':
        data_to_train = (uscat_all.real-mean) / std
        del uscat_all
    elif network_type == 'complexnet':
        u_fft = sfft.fftshift(sfft.fft2(uscat_all),axes=(1,2))
        
        del uscat_all
        data_to_train = np.concatenate(((u_fft.real[:, None, :, :]-mean_r)/std_r,
                                        (u_fft.imag[:, None, :, :]-mean_i)/std_i), axis=1)
        del u_fft
    
    kh = np.shape(data_to_train)[1]
    print(np.shape(data_to_train))


    dataset = torch.utils.data.TensorDataset(
        torch.tensor(data_to_train, dtype=data_type),
        torch.tensor(coefs_all, dtype=data_type)
    )   
    del data_to_train
    logger.info("successfully load training data, ndata {:3}, time: {:.1f}s".format(ndata, time.time() - start_time))
    
    tgt_valid = torch.tensor(tgt_valid, dtype=data_type)
    coef_val = torch.tensor(coef_val, dtype=data_type)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_cfg["batch_size"],
        pin_memory=torch.cuda.is_available()
    )
    loss_fn = nn.MSELoss()
    epoch = train_cfg["epoch"]
    ep_loss = []
    def train(model, device, train_loader, optimizer, epoch, scheduler, model_dir):
        train_logger = logger.getChild("Train Epoch")
        final_error_rel = 1.0
        final_error_abs = 10.0
        k = 0
        max_k = kh
        for e in range(epoch+1):            
            if train_cfg["save_every_nepoch"] > 0 and e % train_cfg["save_every_nepoch"] == 1 and e > 1 and k < max_k:
                #filename = os.path.join(model_dir, "model_"+str(e-1)+".pt")
                #resume(model,optimizer,filename)
                k+=1 
            n_loss = 0
            current_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                #data = (data.to(device)).type(data_type)
                k_data = data[:,k:k+1,:,:]
                k_data = (k_data.to(device)).type(data_type)
                target = (target.to(device)).type(data_type)
                optimizer.zero_grad()
                output = model(k_data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                n_loss += 1
                current_loss += loss.item()
            if e % train_cfg["valid_freq"] == 0:
                coef_pred = model(tgt_valid[:,k:k+1,:,:].to(device))
                loss_train = current_loss / n_loss
                ep_loss.append([e,loss_train])
                loss_val = loss_fn(coef_pred, coef_val.to(device)).item()
                diff = torch.norm(coef_pred.cpu() - coef_val, dim=1).detach().numpy()
                error_rel = np.mean(diff / norm_coef)
                final_error_rel = np.minimum(final_error_rel, error_rel)
                error_abs = np.mean(diff)
                final_error_abs = np.minimum(final_error_abs, error_abs)
                train_logger.info('{:3}, train Loss: {:.6f}, val loss: {:.6f}, rel err: {:.4f}, abs err: {:.4f}, time: {:.1f}s'.format(
                    e, loss_train, loss_val, error_rel, error_abs, (time.time() - start_time))
                )
                writer.add_scalar('loss_train', loss_train, e)
                writer.add_scalar('loss_val', loss_val, e)
                writer.add_scalar('log_log_loss_train', np.log(loss_train), np.log(e+1)*1000)
                writer.add_scalar('log_log_loss_val', np.log(loss_val), np.log(e+1)*1000)
            if train_cfg["save_every_nepoch"] > 0 and e % train_cfg["save_every_nepoch"] == 0 and e > 0:
                torch.save(model.state_dict(), os.path.join(model_dir, "model_"+str(e)+".pt"))
                #checkpoint(model,optimizer,os.path.join(model_dir, "model_"+str(e)+".pt"))
            scheduler.step()
        logger.info('final rel err {:.4f}, abs err {:.4f}'.format(final_error_rel, final_error_abs))
        final_filename = r"C:\Users\blast\Downloads\inverse-obstacle-scattering2d-dev_dl\inverse-obstacle-scattering2d-dev_dl\learn\results\result_star3_kh10.txt"
        final_results = open(final_filename,"a+")
        final_results.write('nc: 10, ndata: {:03d}, final rel err {:.4f}, abs err {:.4f}\n'.format(total_data,final_error_rel, final_error_abs))
        final_results.write(str(ep_loss))
        final_results.write(" \n")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if network_type == 'convnet':
        model = network.ConvNet(data_cfg, train_cfg)
    elif network_type == 'complexnet':
        model = network.ComplexNet(data_cfg, train_cfg)
    model.type(data_type)
    if args.retrain:
        logger.info("retrain model %s", args.retrain)
        model.load_state_dict(torch.load(os.path.join(args.dirname, args.retrain), map_location=device))
    model = model.to(device)
    
    if train_cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=train_cfg["momentum"])
    elif train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    
    train(model, device, train_loader, optimizer, epoch, scheduler, model_dir)
    for k in range(kh):
        coef_pred = model(tgt_valid[:,k:k+1,:,:].to(device))

    writer.close()
    scipy.io.savemat(
        os.path.join(args.dirname, "valid_predby_{}.mat".format(args.model_name)),
        {
            "coef_val": coef_val.numpy().astype('float64'),
            "coef_pred": coef_pred.detach().cpu().numpy().astype('float64'),
            "cfg_str": valid_data["cfg_str"][0]
        }
    )
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    

if __name__ == '__main__':
    main()



    