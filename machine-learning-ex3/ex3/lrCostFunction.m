function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Cost Function.
h = sigmoid(X * theta);

diff = (-y' * log(h) - (1 - y') * log(1 - h));
J = diff  / m;
% Regulate Cost.
tmp_theta = [0 ; theta(2 : end)];
reg_cost = (lambda / (2 * m)) * (tmp_theta' * tmp_theta);
J =  J + reg_cost;

% Gradient
grad = (X' * (h - y)) / m;
% Regulate Gradient.
reg = ((lambda / m) * tmp_theta);
grad = grad + reg;

end