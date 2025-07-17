load("compare_8_9_10_v2_error_array.mat")


% ---------------------------------------------------------------
% Example script to create a new figure for each noise-level/error-type combo
% with 3 datasets in error_array of size [3, 3, 100, 4].
%
%   1) Dataset:      1='8-10', 2='9-10', 3='10'
%   2) Noise level:  3 levels (0, 0.5, 1)
%   3) Iterations:   100
%   4) Error type:   4 types
% ---------------------------------------------------------------
globalMin = min(error_array(:));    
globalMax = max(error_array(:));
% Define labels
datasets    = {'8-10', '9-10','10'};
noiseLevels = [0, 0.5, 1];
errorTypes  = {'Chamfer Predicted', ...
               'Chamfer Refined',   ...
               'L2 Predicted',      ...
               'L2 Refined'};

% Loop over each error type
for e = 1:4
    
    % Loop over each noise level
    for n = 1:3
        
        % Extract the data for the current error type 'e' and noise level 'n'
        % error_array has dimensions: [dataset, noiseLevel, iteration, errorType]
        % This extraction will yield a 3 x 100 matrix (3 datasets, 100 iterations).
        dataMatrix = squeeze(error_array(:, n, :, e));  % shape => [3, 100]
        
        % Transpose so that each column in boxplot is one dataset
        % after transpose => dataMatrix is [100, 3]
        dataMatrix = dataMatrix';
        
        % Create a new figure for this combination
        figure('Name','Error Boxplots','NumberTitle','off');
        
        % Create the boxplot of the three datasets
        boxplot(dataMatrix, 'Labels', datasets);
        ylim([globalMin, globalMax])
        % Format the plot
        xlabel('Dataset');
        ylabel('Error Value');
        titleStr = sprintf('Error Type: %s, Noise Level = %.1f', ...
                           errorTypes{e}, noiseLevels(n));
        title(titleStr);
        
        % Create folder if it doesn't exist yet
        folderName = 'SavedFigures_v3';
        if ~exist(folderName, 'dir')
            mkdir(folderName);
        end
        
        % Construct a filename using your loop variables (e for error type, n for noise level)
        % For example: "ErrorType_1_NoiseLevel_0.5.png"
        fileName = sprintf('ErrorType_%d_NoiseLevel_%g.png', e, noiseLevels(n));
        
        % Save the figure in the newly-created folder
        saveas(gcf, fullfile(folderName, fileName));
    end
end

