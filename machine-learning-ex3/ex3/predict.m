function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
alpha_1 = [ones(m, 1) X];
z_2 = alpha_1 * Theta1';
alpha_2 = sigmoid(z_2);
alpha_2 = [ones(size(alpha_2, 1), 1) alpha_2];
z_3 = alpha_2 * Theta2';
alpha_3 = sigmoid(z_3);

[val, index] = max(alpha_3, [], 2);
p = index;

end
