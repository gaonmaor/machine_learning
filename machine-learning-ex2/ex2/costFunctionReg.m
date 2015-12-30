function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Cost Function.
z = X * theta;
h = sigmoid(z);
diff = (-y.*log(h) - (1 - y).*log(1 - h));
J = (diff' * ones(m, 1)) / m;
% Regulate Cost.
reg_cost = ((lambda / (2 * m)) * (theta(2:end)' * theta(2:end)));
% reg_cost(1) = 0;
J = J + reg_cost;

% Gradient
grad = (X' * (h - y)) / m;
% Regulate Gradient.
reg = (lambda / m) * theta;
reg(1) = 0;
grad = grad + reg;

end