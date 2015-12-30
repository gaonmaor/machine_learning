function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

# Cost Function.
z = X * theta;
h = sigmoid(z);
diff = (-y.*log(h) - (1 - y).*log(1 - h));
J = (diff' * ones(m, 1)) / m;

# Gradient
grad = (X' * (h - y)) / m;

end
