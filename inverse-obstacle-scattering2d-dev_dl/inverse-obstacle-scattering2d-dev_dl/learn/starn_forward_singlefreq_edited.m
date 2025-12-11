function starn_forward_singlefreq(mat_id)
% Usage:
% mat_id = 0 : Run Validation Set
% mat_id = 1 : Run Training Files 1-200
% mat_id = 2 : Run Training Files 201-400
% mat_id = 3 : Run Training Files 401-600
% mat_id = 4 : Run Training Files 601-800

close all
% Don't clear 'mat_id' as it is an input argument
clearvars -except mat_id 
addpath('./REU')
tic

% --- CONFIGURATION ---
cfg_path = './configs/nc20.json';
% You can uncomment logic here if you want to switch configs based on ID
% if mat_id == 0; cfg_path = ...; end

cfg_str = fileread(cfg_path);
cfg = jsondecode(cfg_str);

% --- CRITICAL FIX: PARALLEL POOL WITH LOCAL STORAGE ---
% This prevents the Lustre file locking hang
pool = gcp('nocreate');
if isempty(pool)
    c = parcluster('local');
    
    % Use local node temp directory
    local_tmp = getenv('TMPDIR');
    if isempty(local_tmp); local_tmp = '/tmp'; end
    
    job_folder = fullfile(local_tmp, ['matlab_job_' getenv('USER')]);
    if ~exist(job_folder, 'dir'); mkdir(job_folder); end
    c.JobStorageLocation = job_folder;
    
    % Start pool with 16 workers
    pool = parpool(c, 16);
end
fprintf('Parallel pool has %d workers.\n', pool.NumWorkers);
% ------------------------------------------------------

ndata = cfg.ndata;         % Total data (e.g. 80000)
nvalid = cfg.nvalid;
nc = cfg.nc;
kh = cfg.kh;
start_kh = cfg.kh_start;
dk = cfg.dk;
n = max(300, 50*nc);
ndata_per_mat = cfg.ndata_per_mat; % e.g. 100

% Setup physics parameters
bc = []; bc.type = 'Dirichlet'; bc.invtype = 'o';
src0 = [0.01;-0.12];
opts = []; opts.test_analytic = false; opts.src_in = src0; opts.verbose = false;

% Targets and Directions
r_tgt = cfg.r_tgt; n_tgt = cfg.n_tgt;
t_tgt = 0:2*pi/n_tgt:2*pi-2*pi/n_tgt;
n_dir = cfg.n_dir;
t_dir = 0:2*pi/n_dir:2*pi-2*pi/n_dir;
[t_tgt_grid,t_dir_grid] = meshgrid(t_tgt,t_dir);
t_tgt_grid = t_tgt_grid(:); t_dir_grid = t_dir_grid(:);
xtgt = r_tgt*cos(t_tgt_grid); ytgt = r_tgt*sin(t_tgt_grid);
tgt = [ xtgt'; ytgt'];
sensor_info = []; sensor_info.tgt = tgt; sensor_info.t_dir = t_dir_grid;

% Params
rng(ndata+nvalid)
coefs_val = sample_fc(cfg, nvalid);
NOISE_LVL = cfg.noise_lvl;
khv = start_kh:dk:kh;
nk = length(khv);

% Directory Setup
dirname = ['./data/star' int2str(nc) '_kh' int2str(start_kh) '_' num2str(dk*100) '_' int2str(kh) '_n' int2str(n_tgt) '_' int2str(ndata) '_noise' num2str(NOISE_LVL*100)];
if ~exist(dirname, 'dir'); mkdir(dirname); end
train_data_dir = strcat(dirname, '/train_data');
if ndata>1 && ~exist(train_data_dir, 'dir'); mkdir(train_data_dir); end

save_fcn = @(name, coefs, uscat) save(name, 'coefs', 'uscat');

% =========================================================
% LOGIC SPLIT
% =========================================================

if nargin == 0 || mat_id == 0
    % --- VALIDATION MODE ---
    fprintf('Running Validation Data Generation...\n');
    uscat_val = zeros(nvalid, nk, n_dir, n_tgt);

    parfor idx=1:nvalid
        coefs = coefs_val(idx, :)';
        src_info = geometries.starn(coefs, nc, n);
        for ik = 1:nk
            kk = khv(ik);
            [mats, ~] = rla.get_fw_mats(kk, src_info, bc, sensor_info, opts);
            fields = rla.compute_fields(kk, src_info, mats, sensor_info, bc, opts);
            uscat_val(idx, ik, :, :) = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
        end
    end
    
    if ndata>1
        fname = strcat(dirname, '/valid_data.mat');
        save(fname, 'coefs_val', 'uscat_val', 'cfg_str', '-v7.3');
        fprintf('Successfully saved the validation data \n');
    end

else
    % --- TRAINING MODE (SPLIT INTO 4 JOBS) ---
    % 1. Calculate the range of files for this specific job ID
    total_files   = floor(ndata / ndata_per_mat); % Should be 800
    num_jobs      = 4; % Splitting into 4 chunks
    files_per_job = ceil(total_files / num_jobs); % Should be 200
    
    start_file_idx = (mat_id - 1) * files_per_job + 1;
    end_file_idx   = min(mat_id * files_per_job, total_files);
    
    fprintf('=== JOB ID %d ===\n', mat_id);
    fprintf('Processing Files: %d to %d\n', start_file_idx, end_file_idx);
    
    % 2. Run PARFOR over the FILES (not inside the file)
    % This ensures all 16 workers are busy processing different files at once
    parfor mat_index = start_file_idx : end_file_idx
        
        % Calculate indices for the data samples inside this file
        data_start = (mat_index-1) * ndata_per_mat + 1;
        data_end   = mat_index * ndata_per_mat;
        
        data_name = [train_data_dir '/train_data_' num2str(data_start) '-' num2str(data_end) '.mat'];
        
        % Check if file exists to avoid overwriting (simple restart logic)
        if ~exist(data_name, 'file')
            
            % Generate random coefficients unique to this batch
            % Note: We must be careful with RNG in parfor. 
            % Ideally, seed based on the unique mat_index to be deterministic
            % rng(mat_index); <--- Enabling this ensures reproducibility per file
            
            % Temporary variables for this worker
            local_coefs = sample_fc(cfg, ndata_per_mat);
            local_uscat = complex(zeros(ndata_per_mat, nk, n_dir, n_tgt));
            
            % Inner loop (Serial execution for the 100 items inside one file)
            for local_idx = 1:ndata_per_mat
                coef = local_coefs(local_idx, :)';
                src_info = geometries.starn(coef, nc, n);
                
                for ik = 1:nk
                    % Generate Noise
                    noise = 1 + NOISE_LVL * rand(n_dir, n_tgt).* exp(2*pi*1i * rand(n_dir, n_tgt));
                    
                    kk = khv(ik);
                    [mats,~] = rla.get_fw_mats(kk, src_info, bc, sensor_info, opts);
                    fields = rla.compute_fields(kk, src_info, mats, sensor_info, bc, opts);
                    uscat_temp = reshape(fields.uscat_tgt, [n_dir, n_tgt]);
                    
                    % Apply noise
                    local_uscat(local_idx, ik, :, :) = uscat_temp .* noise;
                end
            end
            
            % Save Once per file (Optimized)
            % Use parfeval or just save directly (parfor handles simple saves)
            par_save(data_name, local_coefs, local_uscat);
            fprintf('Saved batch: %d\n', mat_index);
        else
            fprintf('Skipping batch %d (File exists)\n', mat_index);
        end
    end
end

toc
end

% Helper function to allow saving inside parfor
function par_save(fname, coefs, uscat)
    save(fname, 'coefs', 'uscat');
end
