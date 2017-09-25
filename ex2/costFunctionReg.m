function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);
scale = 1/m;
regularization_scale = lambda/(2*m);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute cost
hypothesis = sigmoid(X*theta);
sum = 0;
regularization_sum = 0;

for i = 1:m
  first_term = -y(i) * log(hypothesis(i));
  second_term = (1-y(i)) * log(1 - hypothesis(i));
  sum += (first_term - second_term);
endfor
sum *= scale;

for j = 2:n
  regularization_sum += theta(j)^2;
endfor
regularization_sum *= regularization_scale;

J = sum + regularization_sum;

% Compute gradient
for j = 1:n
  grad_sum = 0;
  for i = 1:m
    grad_sum += ( hypothesis(i) - y(i) ) * X(i,j);
  endfor
  grad_sum *= scale;
  
  regularization_sum = 0;
  if (j > 1)
    regularization_sum = (lambda/m) * theta(j);
  endif
  
  grad(j) = grad_sum + regularization_sum;
endfor

% =============================================================

end
