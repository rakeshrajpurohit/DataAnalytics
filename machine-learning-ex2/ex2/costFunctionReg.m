function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%hypothsis function for logistic regression
hypothesis = sigmoid(X*theta);

%to start from theta 2  
shift_theta = theta(2:size(theta));

% new theta vector theta(1) as 0
theta_new = [0;shift_theta];

%regularized cost function Jval and gradient 
J = (-y'*log(hypothesis)-(1-y)'*log(1-hypothesis))/m + (lambda*theta_new'*theta_new/(2*m));
grad = (X'*(hypothesis-y)+lambda*theta_new)/m;



% =============================================================

end
