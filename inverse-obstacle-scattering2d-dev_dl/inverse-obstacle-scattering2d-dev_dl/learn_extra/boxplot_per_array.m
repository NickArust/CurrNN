

load('MATS/error_array_6_40_10_2000_0.mat')
error_array_9_10_10_2000_0 = error_array


load('MATS/error_array_6_40_10_2000_1.mat')
error_array_9_10_10_2000_1 = error_array


%load('MATS/error_array_8_10_10_4000_50.mat')
%error_array_9_10_10_2000_2 = error_array


%load('MATS/error_array_8_10_10_4000_75.mat')
%error_array_9_10_10_2000_3 = error_array


%load('MATS/error_array_8_10_10_4000_1.mat')
%error_array_9_10_10_2000_4 = error_array

%errorArrays = {error_array_9_10_10_2000_0,  error_array_9_10_10_2000_1, error_array_9_10_10_2000_2, error_array_9_10_10_2000_3, error_array_9_10_10_2000_4};
errorArrays = {error_array_9_10_10_2000_0,  error_array_9_10_10_2000_1};


% ----------- (1) Combine multiple 3D arrays into a single 4D array -----------
n = numel(errorArrays);                         % number of datasets
[N_levels, N_iterations, N_types] = size(squeeze(errorArrays{1}));

% Initialize the 4D array => [n, N_levels, N_iterations, N_types]
error_array = zeros(n, N_levels, N_iterations, N_types);

% Fill one slice per dataset
for i = 1:n
    error_array(i, :, :, :) = squeeze(errorArrays{i});
end

% ----------- (2) Compute global min/max for uniform y-axis across all plots -----------
globalMin = 0;
globalMax = max(error_array(:));

% ----------- (3) Define your labels -----------
% Make sure 'datasetNames' matches the number of datasets (n).
% Example:
datasetNames = {'NOISE 0', 'NOISE 0.25', 'NOISE 0.5', 'NOISE 0.75', 'NOISE 1'};
noiseLevels = [0,0.5, 1];
errorTypes  = {'Chamfer Predicted', ...
               'Chamfer Refined',   ...
               'L2 Predicted',      ...
               'L2 Refined'};

% ----------- (4) Create (or check) the main folder and a chosen subfolder -----------
mainFolderName = 'NEW_BOXPLOT_PER_ARRAY';   % Your existing main folder
subFolderName  = '6:04:10 log scale';         % Customize as needed
fullFolderPath = fullfile(mainFolderName, subFolderName);
if ~exist(fullFolderPath, 'dir')
    mkdir(fullFolderPath);
end

% ----------- (5) Loop over error types and noise levels, make plots -----------
for e = 1:numel(errorTypes)          % loop over the 4 error types
    for lvl = 1:numel(noiseLevels)   % loop over the noise levels

        % Extract data across all datasets for this noise level (lvl) & error type (e)
        % shape is [n, N_iterations], then transpose => [N_iterations, n]
        dataMatrix = squeeze(error_array(:, lvl, :, e))'; 

        % Create a figure
        figure('Name','Error Boxplots','NumberTitle','off');
        
        % Boxplot: each column in dataMatrix is one dataset
        boxplot(dataMatrix, 'Labels', datasetNames);
        
        % Apply global y-limits
        ylim([globalMin, globalMax]);
        yscale log
        % Axis labels, title
        xlabel('Dataset');
        ylabel('Error Value');
        titleStr = sprintf('6:0.4:10, ndata 2000  \n Error Type: %s, Noise Level = %.1f', ...
                           errorTypes{e}, noiseLevels(lvl));
        title(titleStr);
        
        % Save figure in the subfolder
        fileName = sprintf('double_ErrorType_%d_NoiseLevel_%g.png', e, noiseLevels(lvl));
        saveas(gcf, fullfile(fullFolderPath, fileName));
        
    end
end
