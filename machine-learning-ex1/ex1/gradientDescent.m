function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%% NOTE: The double %% remarks are for future extention for automaticly 
%%   decress alpha if needed.
%%prev_J = Inf;
for iter = 1:num_iters

    theta = theta - ((alpha * (1 / m)) * (X' * ((X * theta) - y)));
    
    % Ensure that J reduces:
    J = computeCost(X, y, theta);
    
    %%if prev_J < J
    %%  % If the new J is bigger -> we need a smaller alpha.
    %%  alpha = alpha / 3
    %%end
    

    % Save the cost J in every iteration    
    J_history(iter) = J;
    %%prev_J = J;

end

end
