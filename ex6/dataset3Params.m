function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
c1 = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma1 = [0.01 0.03 0.1 0.3 1 3 10 30]';
temp=1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for iter =1:length(c1)
for iter2=1:length(sigma1)
c1_test=c1(iter);
sigma_test=sigma1(iter2);
model = svmTrain(X, y, c1_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
predictions = svmPredict(model, Xval);
prediction_error(iter,iter2) = mean(double(predictions ~= yval));
% =========================================================================
end
end


 [values, row_index]=min(prediction_error);
   [~ ,col] = min(values);
   row = row_index(col);

   C = c1(row);
   sigma = sigma1(col);


end
