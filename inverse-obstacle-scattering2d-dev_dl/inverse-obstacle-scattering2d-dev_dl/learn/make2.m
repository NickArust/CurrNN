% Input: err of size 1 x 5 x 100 x 4  (1 dataset, 5 noise levels, 100 iters, 4 error types)
S = load('t10-1_i10_500.mat')
err = S.error_array
% ---- setup ----
assert(ndims(err)==4 && isequal(size(err,[1 2 3 4]), [1 5 100 4]), 'err must be 1x5x100x4');
nNoise  = size(err,2);
nIters  = size(err,3);
nTypes  = size(err,4);
noise_levels = linspace(0,1,nNoise);

% ---- plot: 1 figure per error type, 5 subplots (one per noise) ----
for et = 1:nTypes
    f = figure('Name',sprintf('ErrorType_%d',et),'Color','w');
    t = tiledlayout(1,nNoise,'TileSpacing','compact','Padding','compact');
    for ni = 1:nNoise
        nexttile;
        x = squeeze(err(1,ni,1:nIters,et));  % 100x1
        boxplot(x,'Labels',{sprintf('%.2f',noise_levels(ni))});
        ylabel('Error');
        title(sprintf('Noise = %.2f', noise_levels(ni)));
        grid on;
    end
    title(t, sprintf('Error Type %d', et));
    
    % ---- optional save ----
     saveas(f, sprintf('figures/boxplot_500_%d.png', et));
end

% ---- (optional) alternate view: one figure per noise, 4 subplots (one per error type) ----
%{
for ni = 1:nNoise
    f = figure('Name',sprintf('Noise_%.2f',noise_levels(ni)),'Color','w');
    t = tiledlayout(1,nTypes,'TileSpacing','compact','Padding','compact');
    for et = 1:nTypes
        nexttile;
        x = squeeze(err(1,ni,1:nIters,et));
        boxplot(x,'Labels',{sprintf('Type %d',et)});
        ylabel('Error');
        title(sprintf('Error Type %d', et));
        grid on;
    end
    title(t, sprintf('Noise = %.2f', noise_levels(ni)));
    % saveas(f, sprintf('boxplot_noise_%02d.png', ni));
end
%}

