% --- load data ---
S = load('t10-1_i10_500.mat');
if isfield(S,'error_array')
    err = S.error_array;
elseif isfield(S,'err')
    err = S.err;
else
    error('Could not find error_array (or err) in the MAT file.');
end

% --- ensure shape 1x5x100x4 ---
sz = size(err);
if isequal(sz, [1 5 100 4])
    % ok
elseif numel(sz)==3 && isequal(sz,[5 100 4])
    err = reshape(err,[1 5 100 4]);
else
    error('err has unexpected size: %s', mat2str(sz));
end

% --- setup ---
nNoise  = size(err,2);
nIters  = size(err,3);
nTypes  = size(err,4);
noise_levels = linspace(0,1,nNoise);

% --- one figure per error type (5 boxplots per figure) ---
for et = 1:nTypes
    % collect data for all noise levels
    data = squeeze(err(1,:,1:nIters,et));   % size 5x100
    data = reshape(data, [nNoise, nIters])'; % size 100x5

    f = figure('Name',sprintf('ErrorType_%d',et),'Color','w');
    boxplot(data, 'Labels', arrayfun(@(x) sprintf('%.2f', x), noise_levels, 'UniformOutput', false));
    xlabel('Noise Level');
    ylabel('Error');
    title(sprintf('Refined Chamfer Error', et));
    grid on;

    % optional: save figure
     saveas(f, sprintf('figures/boxplot_errtype_%d.png', et));
end
