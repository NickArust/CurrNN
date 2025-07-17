load("retrained_jcp_error_array_3_noise.mat")

% ---------------------------------------------------------------
% Example script to create a new figure for each noise-level/error-type combo
% error_array size: [2, 3, 100, 4]
%
%   1) Dataset:      1 = 'retrained', 2 = 'base'
%   2) Noise level:  3 levels (0, 0.5, 1)
%   3) Iterations:   100
%   4) Error types:  4 types
% ---------------------------------------------------------------

% Make sure 'error_array' is in your workspace.

datasets    = {'base', 'retrained'};  % dimension 1
noiseLevels = [0, 0.5, 1];           % dimension 2
errorTypes  = {'Chamfer Predicted', ...
               'Chamfer Refined',   ...
               'L2 Predicted',      ...
               'L2 Refined'};       % dimension 4

% Loop over each error type
for e = 1:4
    
    % Loop over each noise level
    for n = 1:3
        
        % Extract the data for the current error type 'e' and noise level 'n'
        % error_array has dimensions [dataset, noiseLevel, iteration, errorType]
        % This extraction yields a 2 x 100 matrix (2 datasets, 100 iterations).
        dataMatrix = squeeze(error_array(:, n, :, e));  % shape => [2, 100]
        
        % Transpose so that each column becomes a group in boxplot
        % After transpose, dataMatrix is [100, 2], so each column is one dataset
        dataMatrix = dataMatrix';  % shape => [100, 2]
        
        % Create a new figure for this combination
        figure('Name','Error Boxplots','NumberTitle','off');
        
        % Create the boxplot of the two datasets
        boxplot(dataMatrix, 'Labels', datasets);
        
        % Format the plot
        xlabel('Dataset');
        ylabel('Error Value');
        titleStr = sprintf('Error Type: %s, Noise Level = %.1f', ...
                           errorTypes{e}, noiseLevels(n));
        title(titleStr);
        
        % (Optional) Adjust any boxplot properties or formatting here
        %grid on
	% -- after calling figure(...) and finishing your boxplot settings --

	% Create folder if it doesn't exist yet
	folderName = 'SavedFigures';
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

