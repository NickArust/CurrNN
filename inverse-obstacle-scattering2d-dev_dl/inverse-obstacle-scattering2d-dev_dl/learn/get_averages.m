% analyze_errors_by_noise_and_type.m
% Go through MATS/ps_2/*.mat, parse filename metadata,
% and for each file compute, for each noise level (5) and error type (4),
% how many of the 100 iterations have error < 1%.

clear; clc;

baseDir = 'MATS/ps_2';
files   = dir(fullfile(baseDir, '*.mat'));

if isempty(files)
    error('No .mat files found in %s', baseDir);
end

nFiles = numel(files);

% Preallocate results structure
results(nFiles) = struct( ...
    'filename',              '', ...
    'kh_start',              NaN, ...
    'dk_code',               NaN, ...   % e.g. 50 meaning Δk = 0.5
    'kh_end',                NaN, ...
    'n_dir',                 NaN, ...
    'n_data',                NaN, ...
    'noise_label',           NaN, ...
    'model_factor',          NaN, ...
    'index_id',              NaN, ...
    'counts_below_1pct',     [], ...    % 5 x 4 matrix
    'total_per_cell',        100);      % iterations per (noise, error type)

for k = 1:nFiles
    fname    = files(k).name;
    fullpath = fullfile(baseDir, fname);

    % ---- Parse filename ----
    % Example: star10_kh9_50_10_n48_600_noise30_model_2_559.mat
    expr = ['^star(?<nc>\d+)_kh(?<khstart>\d+)_' ...
        '(?<dkcode>\d+)_ (?<khend>\d+)_n(?<ndir>\d+)_' ...
        '(?<ndata>\d+)_noise(?<noisecs>\d+)_model_(?<model>[\d\.]+)' ...
        '(?:_(?<idx>\d+))?\.mat$'];
expr = strrep(expr, '_ ', '_');  % remove accidental space

    tokens = regexp(fname, expr, 'names');
    if isempty(tokens)
        warning('Filename "%s" did not match expected pattern. Metadata will be NaN.', fname);
    end

    % ---- Load file and find the 1x5x100x4 error array ----
    S  = load(fullpath);
    fn = fieldnames(S);

    errArray = [];
    for j = 1:numel(fn)
        val = S.(fn{j});
        if isnumeric(val) && ndims(val) == 4
            sz = size(val);
            if isequal(sz, [1 5 100 4])
                errArray = val;
                break;
            end
        end
    end

    if isempty(errArray)
        warning('No 1x5x100x4 numeric array found in "%s". Skipping.', fname);
        continue;
    end

    % ---- Compute counts below 1% for each noise level and error type ----
    % errArray is 1 x 5 x 100 x 4
    % We want, for each noise (dim2) and error type (dim4), count of iterations (dim3)
    % where error < 0.01.
    mask = (errArray < 1);    % logical, same size

    % Sum along the iteration dimension (3rd)
    % Result: 1 x 5 x 1 x 4 -> squeeze to 5 x 4
    counts = squeeze(sum(mask, 3));   % 5 x 4

    % ---- Fill results ----
    results(k).filename          = fname;
    results(k).counts_below_1pct = counts;

    if ~isempty(tokens)
        results(k).kh_start    = str2double(tokens.khstart);
        results(k).dk_code     = str2double(tokens.dkcode);     % Δk = dk_code / 100
        results(k).kh_end      = str2double(tokens.khend);
        results(k).n_dir       = str2double(tokens.ndir);
        results(k).n_data      = str2double(tokens.ndata);
        results(k).noise_label = str2double(tokens.noisecs);    % e.g. 30 => "30% noise" label for file
        results(k).model_factor= str2double(tokens.model);      % e.g. 2, 1.5, 1.0
if isfield(tokens,'idx') && ~isempty(tokens.idx)
        results(k).index_id = str2double(tokens.idx);
    else
        results(k).index_id = NaN;   % no trailing index in filename
    end    
end
end

% Remove entries where we failed to process the array
valid = ~cellfun(@isempty, {results.counts_below_1pct});
results = results(valid);
valid = arrayfun(@(r) ~isempty(r.counts_below_1pct) && r.n_data == 1000, results);
results = results(valid);
% ---- Example of how to inspect results ----
for k = 1:numel(results)
    fprintf('\nFile: %s\n', results(k).filename);
    fprintf('Counts of iterations with error < 1%% (rows = noise level 1..5, cols = error type 1..4):\n');
    disp(results(k).counts_below_1pct);
end
fid = fopen('results_summary.txt','w');

for k = 1:numel(results)
    fprintf(fid, '\nFile: %s\n', results(k).filename);
    fprintf(fid, 'Counts of iterations with error < 1%% (rows = noise level 1..5, cols = error type 1..4):\n');

    % Write the matrix row-by-row
    fprintf(fid, '%d %d %d %d\n', results(k).counts_below_1pct.');
end

fclose(fid);

