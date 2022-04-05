function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h_theta = X*theta;
for i = 1:m
  sub_h = (h_theta(i) - y(i))^2;
  J = J + sub_h;
endfor
J = J/(2*m)
#h_theta = X*theta;
#sub_h = h_theta.-y;
#sqr_h = sub_h.^2;
#sum_h = sum(sqr_h);
#J = sum_h/(2*m)



% =========================================================================

end