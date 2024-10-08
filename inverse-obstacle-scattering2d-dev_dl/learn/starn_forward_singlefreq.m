% This script generates the data for a star shaped domain, for a fixed
% number of sensors and incident directions where data is available for all
% sensors at each incident direction
function starn_forward_singlefreq(mat_id)
close all
clearvars -except mat_id
tic
if nargin == 0
    fprintf('Runing the code on one server \n')
elseif mat_id == 0
    fprintf('Generating validation data \n')
else
    fprintf(['Generating training data through parallel computing with mat index ' ...
    num2str(mat_id) ', will not generate validation data. \n'])
end

cfg_path = './configs/nc10.json';
data_prefix = '';
cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);
ndata = cfg.ndata;
nvalid = cfg.nvalid;
% max number of wiggles
nc = cfg.nc;
kh = cfg.kh;
n  = max(300, 50*nc);
disp(ndata) 
% Test obstacle Frechet derivative for Dirichlet problem
bc = [];
bc.type = 'Dirichlet';
bc.invtype = 'o';

src0 = [0.01;-0.12];
opts = [];
opts.test_analytic = false;
opts.src_in = src0;
opts.verbose = false;

% set target locations
%receptors (r_{\ell})
r_tgt = cfg.r_tgt;
n_tgt = cfg.n_tgt;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;

% Incident directions (d_{j})
n_dir = cfg.n_dir;
t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;
[t_tgt_grid,t_dir_grid] = meshgrid(t_tgt,t_dir);
t_tgt_grid = t_tgt_grid(:);
t_dir_grid = t_dir_grid(:);
xtgt = r_tgt*cos(t_tgt_grid);
ytgt = r_tgt*sin(t_tgt_grid);
tgt   = [ xtgt'; ytgt'];

sensor_info = [];
sensor_info.tgt = tgt;
sensor_info.t_dir = t_dir_grid;

% parameters 'a'
rng(ndata+nvalid)
coefs_val = sample_fc(cfg, nvalid);


dirname = ['./data/star' int2str(nc) '_kh' int2str(kh) '_n' int2str(n_tgt) '_' int2str(ndata)];
if ~strcmp(data_prefix, '')
    dirname = strcat(dirname, '_', data_prefix);
end
if ndata>1 && ~exist(dirname, 'dir')
    mkdir(dirname)
end
train_data_dir = strcat(dirname, '/train_data');
if ndata>1 && ~exist(train_data_dir, 'dir')
    mkdir(train_data_dir)
end
khv = 1:kh;
nk = length(khv);
if nargin == 0 || mat_id == 0

    uscat_val = zeros(nvalid, nk, n_dir, n_tgt);

    parfor idx=1:nvalid
        coefs = coefs_val(idx, :)';
        src_info = geometries.starn(coefs,nc,n);
        for ik = 1:nk
            kk = khv(ik);
            [mats,~] = rla.get_fw_mats(kk,src_info,bc,sensor_info,opts);
            fields = rla.compute_fields(kk,src_info,mats,sensor_info,bc,opts);
            uscat_val(idx,ik, :, :) = reshape(fields.uscat_tgt,[n_dir, n_tgt]);
        end
    end
    src_info = geometries.starn(coefs_val(1, :)',nc,n);
    figure
    uscat_tgt = squeeze(uscat_val(1,kh, :, :));
    % imagesc(abs(fftshift(fft2(uscat_tgt))))
    imagesc(abs(((uscat_tgt))))
    figure
    hold on
    plot(src_info.xs,src_info.ys,'b.');
    plot(0, 0, 'r*');

    if ndata>1 
        fname = strcat(dirname, '/valid_data.mat');
        save(fname, 'coefs_val', 'uscat_val', 'cfg_str');
        fprintf('Successfully saved the validation data \n')
    end
end
save_fcn = @(name, coefs, uscat) save(name, 'coefs', 'uscat');
ndata_per_mat = cfg.ndata_per_mat;

if nargin == 0
    fprintf('Start to generate training data \n')
    nmat = ndata / ndata_per_mat;
    
    parfor mat_index=1:nmat
        data_start_index = (mat_index-1) * ndata_per_mat + 1;
        data_end_index = mat_index * ndata_per_mat;
        data_name = [train_data_dir '/train_data_' num2str(data_start_index) '-' num2str(data_end_index) '.mat'];
        coefs = sample_fc(cfg, ndata_per_mat);
        
        uscat = complex(zeros(ndata_per_mat, nk,n_dir, n_tgt));
        for local_idx = 1:ndata_per_mat
            coef = coefs(local_idx, :)';
            src_info = geometries.starn(coef,nc,n);
                    for ik = 1:nk
                        kk = khv(ik);
                        [mats,~] = rla.get_fw_mats(kk,src_info,bc,sensor_info,opts);
                        fields = rla.compute_fields(kk,src_info,mats,sensor_info,bc,opts);
                        uscat_temp = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
                        uscat(local_idx,ik,:,:) = uscat_temp;
                    end

        save_fcn(data_name, coefs, uscat);
        end
    end
elseif mat_id >= 1
    data_start_index = (mat_id - 1) * ndata_per_mat + 1;
    data_end_index = mat_id * ndata_per_mat;
    data_end_index = min(data_end_index, ndata);
    fprintf(['Start to generate training data indexed from ' num2str(data_start_index) ...
    ' to ' num2str(data_end_index) '\n'])
    data_name = [train_data_dir '/train_data_' num2str(data_start_index) '-' num2str(data_end_index) '.mat'];
    if ~exist(data_name, 'file')
        rng(mat_id)
        coefs = sample_fc(cfg, ndata_per_mat);
        print(ndata_per_mat)
        uscat = complex(zeros(ndata_per_mat,nk, n_dir, n_tgt));
        for local_idx = 1:ndata_per_mat
            coef = coefs(local_idx, :)';
            src_info = geometries.starn(coef,nc,n);
                    for ik = 1:nk
                        kk = khv(ik);
                        [mats,~] = rla.get_fw_mats(kk,src_info,bc,sensor_info,opts);
                        fields = rla.compute_fields(kk,src_info,mats,sensor_info,bc,opts);
                        uscat_temp = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
                        uscat(local_idx,ik,:,:) = uscat_temp;
        end
        save_fcn(data_name, coefs, uscat);
        end
    end
end
time=toc
end