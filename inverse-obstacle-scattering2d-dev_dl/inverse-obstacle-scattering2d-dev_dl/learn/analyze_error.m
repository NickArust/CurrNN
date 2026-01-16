clear; clc; close all;

% 1. Load the file
filename = 'MATS/chunk/new_loss.mat';
if isfile(filename)
    dataStruct = load(filename);
else
    error('File %s not found.', filename);
end

% 2. Dynamic Variable Detection
% This finds the variable name regardless of what you named it inside the .mat
vars = fieldnames(dataStruct);
varName = vars{1}; 
raw_data = dataStruct.(varName);

fprintf('Loaded variable ''%s'' with size: %s\n', varName, mat2str(size(raw_data)));

% 3. Preprocessing
% Current size: [1, 5, 100, 4] -> [Singleton, Noise, Trials, ErrorTypes]
% Target size: [5, 100, 4]
clean_data = squeeze(raw_data); 

[n_noise, n_trials, n_types] = size(clean_data);

% Generate Labels
noise_labels = arrayfun(@(x) sprintf('Noise %d', x), 1:n_noise, 'UniformOutput', false);
type_labels = arrayfun(@(x) sprintf('Error Type %d', x), 1:n_types, 'UniformOutput', false);

% --- VISUALIZATION (Box Plots) ---
figure('Name', 'Error Analysis', 'Color', 'w', 'Position', [100, 100, 1200, 800]);

for t = 1:n_types
    subplot(2, 2, t);
    
    % Extract data for this error type: Size [5, 100]
    % We transpose to [100, 5] because boxplot treats columns as groups
    group_data = clean_data(:, :, t)'; 
    
    % Create Box Plot
    boxplot(group_data, 'Labels', noise_labels);
    
    title(type_labels{t}, 'FontWeight', 'bold');
    ylabel('Error Magnitude');
    grid on;
    
    % Optional: Log scale if errors vary by orders of magnitude
    % set(gca, 'YScale', 'log'); 
end

sgtitle('Distribution of Errors over 100 Independent Trials');
exportgraphics(gcf, 'error_analysis_results.pdf', 'ContentType', 'vector');
% --- STATISTICAL SUMMARY TABLE ---
fprintf('\n--- Mean Error +/- Std Dev ---\n');
fprintf('%-15s', 'Noise Level');
for t = 1:n_types
    fprintf('%-20s', type_labels{t});
end
fprintf('\n');

for n = 1:n_noise
    fprintf('%-15s', noise_labels{n});
    for t = 1:n_types
        % Extract the 100 trials for this specific noise/type combo
        trials = clean_data(n, :, t);
        
        mu = mean(trials);
        sigma = std(trials);
        
        fprintf('%-20s', sprintf('%.4f +/- %.4f', mu, sigma));
    end
    fprintf('\n');
end
