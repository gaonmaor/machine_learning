function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

# Cost Function.
h = X * theta;
k = 1 / (2 * m);
J = sumsq(h - y) * k;
reg = lambda * k * sumsq(theta(2:end));
J = J + reg;

# Gradient
grad = (X' * (h - y)) / m;
reg = (lambda / m) * theta;
reg(1) = 0;
grad = grad + reg;
grad = grad(:);

end
