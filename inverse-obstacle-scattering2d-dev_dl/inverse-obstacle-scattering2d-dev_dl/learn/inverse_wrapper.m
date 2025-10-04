
function inverse_wrapper(path, slurm_task_id)
% Extract name from path to use for saving
	tic
    disp(["slurm task id = ", slurm_task_id])

    [name, model, ext] = fileparts(path(9:end)) 
    disp(name)
    disp(model)
    model = [model ext]
    disp(model)
    if endsWith(model, "_")
    model = extractBefore(model, strlength(model)); % remove last char
    end
    disp(model)
    num_noises = 5;
    noise_levels = linspace(0,1,num_noises)
    
    num_tweaks = 100;
    tweak_factor = 0.025;
    
    %error_array = zeros(1, num_noises, num_tweaks,5);
    error_array = zeros(1, num_noises, num_tweaks,4);
    
    curr_path = [name, '/', model]
    for j=1:num_noises
        noise_lvl = noise_levels(j);
        for k=1:num_tweaks
            disp(["folder = ", curr_path])
            disp(["noise = ", num2str(j)])
            disp(['k = ', num2str(k)])
            errors = nicks_inverse(path, noise_lvl, k);
            error_array(1,j,k,1) = errors(1);
            error_array(1,j,k,2) = errors(2);
            error_array(1,j,k,3) = errors(3);
            error_array(1,j,k,4) = errors(4);
            %error_array(1,j,k,5) = errors(5);
    	    
        end
    end

save(['MATS/ps_2/', name , '_', model, '.mat'], 'error_array', '-v7.3');

time = toc
end
