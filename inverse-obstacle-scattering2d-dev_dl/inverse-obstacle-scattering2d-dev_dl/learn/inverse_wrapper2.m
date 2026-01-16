
function inverse_wrapper(path)
% Extract name from path to use for saving
	tic

    num_noises = 5;
    noise_levels = linspace(0,1,num_noises)
    
    num_tweaks = 100;
    tweak_factor = 0.025;
    
    %error_array = zeros(1, num_noises, num_tweaks,5);
    error_array = zeros(1, num_noises, num_tweaks,4);
    
    curr_path = ['data/', path]
    for j=1:num_noises
        noise_lvl = noise_levels(j);
        for k=1:num_tweaks
            disp(["folder = ", curr_path])
            disp(["noise = ", num2str(j)])
            disp(['k = ', num2str(k)])
            errors = nicks_inverse(curr_path, noise_lvl, k);
            error_array(1,j,k,1) = errors(1);
            error_array(1,j,k,2) = errors(2);
            error_array(1,j,k,3) = errors(3);
            error_array(1,j,k,4) = errors(4);
            %error_array(1,j,k,5) = errors(5);
    	    
        end
    end

save(['MATS/chunk/unbalanced.mat'], 'error_array', '-v7.3');

time = toc
end
