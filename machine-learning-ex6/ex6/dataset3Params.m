function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
vals = [0.01 0.03 0.1 0.3 1 3 10 30];
min_pred = inf; 
for i = 1 : numel(vals)
  for j = 1 : numel(vals)
    tmp_C = vals(i);
    tmp_sigma = vals(j);
    model = svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));
    predictions = svmPredict(model, Xval);
    pred = mean(double(predictions ~= yval));
    if (pred < min_pred)
      min_pred = pred;
      C = tmp_C;
      sigma = tmp_sigma;
    end
  end
end

end
