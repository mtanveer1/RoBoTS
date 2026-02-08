function [uu1, uu2, bb1, bb2, Accuracy, train_time] = RoBoSS_TSVM_function( ...
    xTrain, yTrain, xTest, yTest, a, b, C, c, sigma)
% RoBoSS_TSVM_function.m
% ---------------------------------------------------------
% RoBoSS-TSVM (Nonlinear, RBF kernel) for binary classification.
%
% Inputs:
%   xTrain : (m x d) training features
%   yTrain : (m x 1) training labels in {+1, -1}
%   xTest  : (t x d) test features
%   yTest  : (t x 1) test labels in {+1, -1}
%   a, b   : RoBoSS loss parameters
%   C      : structural regularization parameter
%   c      : loss regularization parameter
%   sigma  : RBF kernel width
%
% Outputs:
%   uu1, bb1 : parameters for +1 hypersurface (u_+, b_+)
%   uu2, bb2 : parameters for -1 hypersurface (u_-, b_-)
%   Accuracy : test accuracy (0..1)
%   train_time : training time in seconds
%
% Notes:
%   - Bias is handled by augmenting kernel matrices with a column of ones.
%   - Prediction uses distance-to-two-hyperplanes rule:
%       label = sign( |K(x,Train)*[u_+;b_+]| - |K(x,Train)*[u_-;b_-]| )
% ---------------------------------------------------------

% Separate class-wise training data
A = xTrain(yTrain == 1, :);
B = xTrain(yTrain == -1, :);

% Kernel type: 3 = RBF (see kernelfunction.m)
kernel_type = 3;

mTrain = size(xTrain, 1);
mA     = size(A, 1);
mB     = size(B, 1);
mTest  = size(xTest, 1);

% ---- Build kernel blocks ----
KA = zeros(mA, mTrain);
KB = zeros(mB, mTrain);
KTest = zeros(mTest, mTrain);

for i = 1:mA
    for j = 1:mTrain
        KA(i,j) = kernelfunction(kernel_type, A(i,:), xTrain(j,:), sigma);
    end
end

for i = 1:mB
    for j = 1:mTrain
        KB(i,j) = kernelfunction(kernel_type, B(i,:), xTrain(j,:), sigma);
    end
end

for i = 1:mTest
    for j = 1:mTrain
        KTest(i,j) = kernelfunction(kernel_type, xTest(i,:), xTrain(j,:), sigma);
    end
end

% ---- Augment with bias column ----
KA = [KA, ones(mA, 1)];
KB = [KB, ones(mB, 1)];
KTest = [KTest, ones(mTest, 1)];

n1 = size(KA, 2); % = mTrain + 1
n2 = size(KB, 2); % = mTrain + 1

% ---- Iterative fixed-point updates ----
Nmax = 50;
tol  = 1e-6;

tic;

% Gradient-like mapping for Z1 (positive hyperplane variables)
    function g = gfun1(Z1)
        R = zeros(n1, mB);
        for jj = 1:mB
            s = 1 + KB(jj,:) * Z1;
            if s > 0
                % elementwise scalar * vector
                R(:, jj) = c .* (KB(jj,:).') .* ( -b * a^2 * s * exp(-a * s) );
            else
                R(:, jj) = 0;
            end
        end
        g = sum(R, 2);
    end

% Gradient-like mapping for Z2 (negative hyperplane variables)
    function g = gfun2(Z2)
        S = zeros(n2, mA);
        for jj = 1:mA
            s = 1 - KA(jj,:) * Z2;
            if s > 0
                S(:, jj) = -c .* (KA(jj,:).') .* ( -b * a^2 * s * exp(-a * s) );
            else
                S(:, jj) = 0;
            end
        end
        g = sum(S, 2);
    end

% Precompute matrices used in the closed-form style update
% M1 = (I - KA' * (C I + KA KA')^{-1} * KA)
% M2 = (I - KB' * (C I + KB KB')^{-1} * KB)
I1 = eye(n1);
I2 = eye(n2);

M1 = I1 - KA' * ((C * eye(mA) + KA * KA') \ KA);
M2 = I2 - KB' * ((C * eye(mB) + KB * KB') \ KB);

% Solve for Z1
Z1_old = zeros(n1, 1);
for t = 1:Nmax
    g1 = gfun1(Z1_old);
    Z1 = (1 / C) * (M1 * g1);
    if norm(Z1 - Z1_old) < tol
        break;
    end
    Z1_old = Z1;
end

% Solve for Z2
Z2_old = zeros(n2, 1);
for t = 1:Nmax
    g2 = gfun2(Z2_old);
    Z2 = -(1 / C) * (M2 * g2);
    if norm(Z2 - Z2_old) < tol
        break;
    end
    Z2_old = Z2;
end

train_time = toc;

% Extract (u, b)
uu1 = Z1(1:end-1);
bb1 = Z1(end);

uu2 = Z2(1:end-1);
bb2 = Z2(end);

% ---- Prediction ----
u1 = [uu1; bb1];
u2 = [uu2; bb2];

d1 = abs(KTest * u1);
d2 = abs(KTest * u2);

score = d1 - d2;
pred = ones(size(score));
pred(score >= 0) = -1;  % if d1 >= d2 => closer to negative => -1
pred(score < 0)  =  1;

Accuracy = mean(pred == yTest);

end
