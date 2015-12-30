function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% hidden_layer_size:  25
% input_layer_size:  400
%Theta1: 25x401
%Theta2: 10x26
% X: 5000:400
% y: 5000:1
% Setup some useful variables
% m: 5000 X: 5000x400
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
%Theta1_grad: 25x401
Theta1_grad = zeros(size(Theta1));
%Theta2_grad: 10x26
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% Used for converting y to a matrix of boolean.
% m: 5000
m = size(X, 1);
A = 1:num_labels;
% Y: 5000x10
Y = y == A;

% Cost Function.
% a1: 5000x401
a1 = [ones(m , 1) X];
% z2: 5000x401 X 401x25 = 5000x25
z2 = a1 * Theta1';
% g2: 5000x25 
g2 = sigmoid(z2);
% a2: 5000x26
a2 = [ones(m , 1) g2];
% z3: 5000x26 X 26x10 = 5000x10
z3 = a2 * Theta2';
% a3: 5000x10
a3 = sigmoid(z3);
% h: 5000x10
h = a3;
% diff: 5000x10
diff = (-Y .* log(h) - (1 - Y) .* log(1 - h));
J = sum(diff(:)) / m;

% Regulate Cost.
% tmp_Theta1: 25x401
tmp_Theta1 = [zeros(hidden_layer_size, 1)  Theta1(:, 2 : end)];
% tmp_Theta1: 10x26
tmp_Theta2 = [zeros(num_labels, 1)  Theta2(:, 2 : end)];
J = J + (lambda / (2 * m)) * (sum((tmp_Theta1 .* tmp_Theta1)(:)) + 
                                 sum((tmp_Theta2 .* tmp_Theta2)(:)));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delta_3 = h - Y;
delta_2 = (delta_3 * Theta2(:, 2 : end));
delta_2 = delta_2 .* sigmoidGradient(z2);
Theta1_grad = (delta_2' * a1) / m;
Theta2_grad = (delta_3' * a2) / m;

% Gradient
%%grad = (X' * (h - y)) / m;
% Regulate Gradient.
%%reg = ((lambda / m) * tmp_theta);
%%grad = grad + reg;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda * Theta1(:, 2:end) / m);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda * Theta2(:, 2:end) / m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
