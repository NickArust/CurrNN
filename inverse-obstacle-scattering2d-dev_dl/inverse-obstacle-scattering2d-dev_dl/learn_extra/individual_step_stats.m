% Create output directory if it doesn't exist
output_dir = 'boxplot_figures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Define the pairs of files to compare
base_files = {'o_10_jcp_error_array.mat', 'o_25_jcp_error_array.mat', 'o_5_jcp_error_array.mat'};
step_files = {'step_10_jcp_error_array.mat', 'step_25_jcp_error_array.mat', 'step_50_jcp_error_array.mat'};
pair_names = {'o1-step10', 'o25-step25', 'o5-step50'};
error_types = {'chamfer-pred', 'chamfer-refined', 'l2-pred', 'l2-refined', 'Type 5'};
noise_levels = 1:10;
datasets = {'nc10', 'nc15'};

% First pass: find global max value
global_max = 0;
for pair_idx = 1:length(pair_names)
    % Load files
    base_data = load(base_files{pair_idx});
    step_data = load(step_files{pair_idx});
    
    % Get array names
    base_varname = fieldnames(base_data);
    step_varname = fieldnames(step_data);
    
    % Update global max
    global_max = max([global_max, ...
                     max(base_data.(base_varname{1})(:)), ...
                     max(step_data.(step_varname{1})(:))]);
end

% Second pass: create plots with consistent y-limits
for noise_idx = 1:length(noise_levels)
    for error_type = 1:5
        for dataset_idx = 1:2
            for pair_idx = 1:length(pair_names)
                % Create new figure with invisible display
                fig = figure('Visible', 'off', 'Position', [100 100 600 400]);
                
                % Load the base and step files for this pair
                base_data = load(base_files{pair_idx});
                step_data = load(step_files{pair_idx});
                
                % Get the actual array names
                base_varname = fieldnames(base_data);
                step_varname = fieldnames(step_data);
                base_array = base_data.(base_varname{1});
                step_array = step_data.(step_varname{1});
                
                % Extract data for current noise level, error type, and dataset
                base_values = squeeze(base_array(dataset_idx, noise_idx, :, error_type));
                step_values = squeeze(step_array(dataset_idx, noise_idx, :, error_type));
                
                % Create box plot
                boxplot([base_values, step_values], 'Labels', {'Base', 'Step'});
                
                % Set y-limits
                ylim([0, global_max]);
                
                % Add labels and title
                ylabel('Error Value');
                xlabel('Method');
                title(sprintf('%s - %s\n %s - Noise Level %d', ...
                    upper(datasets{dataset_idx}), pair_names{pair_idx}, error_types{error_type}, noise_idx));
                
                % Save figure
                filename = sprintf('%s_%s_%s_NoiseLevel%d.png', ...
                    datasets{dataset_idx}, pair_names{pair_idx}, error_types{error_type}, noise_idx);
                saveas(fig, fullfile(output_dir, filename));
                
                % Close figure
                close(fig);
            end
        end
    end
end

fprintf('All figures have been saved to the "%s" directory.\n', output_dir);
