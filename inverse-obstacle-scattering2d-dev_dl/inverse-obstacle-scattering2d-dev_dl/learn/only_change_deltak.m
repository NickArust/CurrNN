folder = 'MATS/ps_2';
files = dir(fullfile(folder, 'star*.mat'));


for fixed_k0 = 6:9
	for fixed_npoints = [100, 200, 300, 400, 500, 600, 1000]
		for fixed_noise = [0, 10, 20, 30]
			for noise_idx = 1:5
%fixed_k0 = 9;
%fixed_npoints = 1000;
%fixed_noise = 0;
disp(fixed_k0)
disp(fixed_npoints)

disp(fixed_noise)
disp(noise_idx)
results = [];
for file = files'
    name = file.name;
    
    % Extract parameters from filename
    tokens = regexp(name, ...
    'star(\d+)_kh(\d+)_([\d]+)_([\d]+)_n(\d+)_([\d]+)_noise(\d+)_model_([0-9]+(?:\.[0-9]+)?)', ...
    'tokens');

    if isempty(tokens), continue; end
    params = str2double(tokens{1});
    
    k0 = params(2);
    step = params(3)/100; % Assuming 20 means 0.2
    kend = params(4);
    npoints = params(6);
    noise = params(7);
    model = params(8);
    if k0 ~= fixed_k0 || npoints ~= fixed_npoints || noise ~= fixed_noise || model ~= floor(model)
        continue;
    end
 %   disp(name)
    data = load(fullfile(folder, name));
    matrix = data.error_array;  % Adjust to actual variable name
 % noise_idx = 1;  % first noise level

% ... inside your file loop, after loading `matrix` ...
% Mean & std across runs at chosen noise, error type 4
runs_vec = squeeze(matrix(1, noise_idx, :, 2));  % -> 100x1
mean_err = mean(runs_vec);
std_err  = std(runs_vec);

results = [results; struct('step', step, ...
                           'mean_err', mean_err, ...
                           'std_err',  std_err, ...
                           'model', model)];
  
    % Compute mean error across runs
%    mean_errors = squeeze(mean(matrix, 3));  % 5 x 4
%results = [results; struct('step', step, 'mean_errors', mean_errors, 'model', model)];    
end

results.model


% Assuming 'results' is an array of structs with fields 'step' and 'mean_errors'

numResults = length(results);
step_sizes = zeros(1, numResults);
mean_error_values = zeros(1, numResults);
std_error_values = zeros(1, numResults);
%{
for i = 1:numResults
    step_sizes(i) = results(i).step;
    errors = results(i).mean_errors(:,4);  % 5 x 4 matrix

    % Flatten and compute mean and std
    all_errors = errors(:);
    mean_error_values(i) = mean(all_errors);
    std_error_values(i) = std(all_errors);
end
% Get unique model values
models = unique([results.model]);
colors = lines(length(models));  % Generate distinct colors

fig = figure('Visible', 'off');
hold on;

for m = 1:length(models)
    model_val = models(m);
    color = colors(m, :);
    
    % Filter results for this model
    model_results = results([results.model] == model_val);
    
    % Extract step sizes and error stats
    step_sizes = [model_results.step];
    mean_error_values = arrayfun(@(r) mean(r.mean_errors(:,2)), model_results);
    std_error_values = arrayfun(@(r) std(r.mean_errors(:,2)), model_results);
    
    % Plot with assigned color
    errorbar(step_sizes, mean_error_values, std_error_values, 'o-', ...
        'Color', color, 'LineWidth', 1.5, 'CapSize', 8, ...
        'DisplayName', sprintf('Model %.1f', model_val));
end

xlabel('Step Size');
ylabel('Mean Error ± Std Dev (Error Type 4)');
title('Effect of Step Size on Error Metrics by Model');
legend('show');
grid on;
saveas(fig, sprintf('step_size_error_by_model_%d_%d_noise_%d.png', ...
    fixed_k0, fixed_npoints, fixed_noise));

close(fig);



% Plotting
figure;
errorbar(step_sizes, mean_error_values, std_error_values, 'o-', 'LineWidth', 1.5, 'CapSize', 8);
xlabel('Step Size');
ylabel('Mean Error ± Std Dev');
title('Effect of Step Size on Error Metrics');
grid on;

% Plotting without displaying
fig = figure('Visible', 'off');  % Create figure but don't show it

errorbar(step_sizes, mean_error_values, std_error_values, 'o-', 'LineWidth', 1.5, 'CapSize', 8);
xlabel('Step Size');
ylabel('Mean Error ± Std Dev');
title('Effect of Step Size on Error Metrics');
grid on;

% Save the figure


saveas(fig, 'step_size_error_analysis_7_500.png');  % Saves as PNG
% Or use: print(fig, 'step_size_error_analysis', '-dpng') for more control

close(fig);  % Close the figure to free memory

%}

fig = figure('Visible', 'off'); hold on;
models = unique([results.model]);
colors = lines(length(models));

for m = 1:length(models)
    model_val = models(m);
    color = colors(m, :);

    model_results = results([results.model] == model_val);

    step_sizes        = [model_results.step];
    mean_error_values = [model_results.mean_err];
    std_error_values  = [model_results.std_err];

    errorbar(step_sizes, mean_error_values, std_error_values, 'o-', ...
        'Color', color, 'LineWidth', 1.5, 'CapSize', 8, ...
        'DisplayName', sprintf('Model %.1f', model_val));
end

xlabel('Step Size');
ylabel('Mean Refied Chmafer Error ± Std Dev');
title_str = sprintf('k_s = %d, ndata = %d, noise = %d, inversion noise = %d', fixed_k0, fixed_npoints, fixed_noise, noise_idx)
title(title_str);
legend('show'); grid on;
saveas(fig, sprintf('figure_step_size/chamfer/step_size_error_by_model_%d_%d_dgNoise_%d_iNoise_%d.png', ...
    fixed_k0, fixed_npoints, fixed_noise, noise_idx));
close(fig);
end
end
end
end
