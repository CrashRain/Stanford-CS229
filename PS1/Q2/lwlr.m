function y = lwlr(X_train, y_train, x, tau)

epsilon = 1e-5;
max = 50;
lambda = 0.0001;

m = size(X_train, 1); % row number of X train
n = size(X_train, 2); % column number of X train

assert(m == size(y_train, 1));

w = nan(m, 1);
for i = 1 : m
    assert(size(x, 1) == size(X_train(i, :)', 1));
    w(i) = exp(-norm(x - X_train(i, :)') ^ 2 / (2 * tau ^ 2));
end

% is equal to
% w = exp(-sum((X_train - repmat(x', m, 1)) .^ 2, 2) / (2 .* tau .^ 2));

theta_old = nan(n, 1);
theta = zeros(n, 1);

for i = 1 : max
    if (norm(theta_old - theta) < epsilon)
        break
    else
        theta_old = theta;
    end
    
    h = 1.0 ./ (1.0 + exp(-(X_train * theta))); % select the logistic model
    D = diag(-w .* h .* (1.0 - h));
    H = X_train' * D * X_train - lambda * eye(n); % Hessian matrix
    z = w .* (y_train - h);
    delta = X_train' * z - lambda * theta;
    theta = theta - H \ delta;
end

if (i >= max)
    disp('Regression failed');
end

y = 1.0 ./ (1.0 + exp(-(x' * theta))) > 0.5;
end