    M = 10;
    % Generate c0 from a uniform distribution Unif(1, 1.2)
    c0 = 1 + 0.2 * rand();
    
    % Initialize the c vector with zeros
    c = zeros(2*M + 1, 1); % Plus 1 for c0
    
    % Set the first element as c0
    c(1) = c0;
    
    % Generate data for each j from 1 to M
    for j = 1:M
        r = 0.1 * rand(); % r ~ Unif(0, 0.1)
        theta = 2 * pi * rand(); % theta ~ Unif(0, 2*pi)
        
        % Convert polar to Cartesian coordinates
        c(j+1) = r * cos(theta); % cj
        c(j+M+1) = r * sin(theta); % cj+M
    end

    rng('shuffle')
    tweak_noise = 0.025 * randn(size(c));
    coef = tweak_noise + c;

    save('coef.mat', 'coef');
