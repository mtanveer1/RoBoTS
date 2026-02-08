%% Main_RoBoSS_TSVM.m
% RoBoSS-TSVM (Nonlinear)
% ---------------------------------------------------------
% Supported input formats:
%   (A) .txt : numeric text file, last column is label
%   (B) .mat : either
%       1) a single matrix variable where last column is label, OR
%       2) variables like (X,y) / (data) / (Data)
%
% Expected labels (binary):
%   - {1, 0} or {1, -1}
% This demo converts label "0" to "-1" to match TSVM convention.
%
% Output:
%   - Prints test accuracy
%
% Notes for full experiments:
%   Typical tuning ranges used in papers:
%     C     = 10.^(-6:2:6)
%     c     = 10.^(-6:2:6)
%     sigma = 10.^(-6:2:6)
%     a     = 0.1:0.2:5.1
%     b     = 0.5:0.5:1.5
%   Evaluation setup:
%     - k-fold CV (e.g., 4-fold) + mean/std reporting
% ---------------------------------------------------------

close all; clear; clc;
rng(42);

%% -------------------- User: set dataset path --------------------


data_path = fullfile(pwd,'congressional_voting.mat');

%% -------------------- Load data (.txt or .mat) --------------------
[~, ~, ext] = fileparts(data_path);
ext = lower(ext);

switch ext
    case '.txt'
        raw = importdata(data_path);
        if isstruct(raw), raw = raw.data; end
        data = raw;

        [m, n] = size(data);
        X = data(:, 1:n-1);
        y = data(:, n);

    case '.mat'
        S = load(data_path);
        vars = fieldnames(S);

        % Case 1: explicit X and y
        if isfield(S, 'X') && isfield(S, 'y')
            X = S.X; y = S.y;

        elseif isfield(S, 'x') && isfield(S, 'y')
            X = S.x; y = S.y;

            % Case 2: explicit data matrix
        elseif isfield(S, 'data')
            data = S.data;
            [m, n] = size(data);
            X = data(:, 1:n-1);
            y = data(:, n);

        elseif isfield(S, 'Data')
            data = S.Data;
            [m, n] = size(data);
            X = data(:, 1:n-1);
            y = data(:, n);

        else
            % Case 3: try to find a single matrix variable
            % (common when .mat stores one variable only)
            found = false;
            for k = 1:numel(vars)
                V = S.(vars{k});
                if isnumeric(V) && ismatrix(V) && size(V,2) >= 2
                    data = V;
                    [m, n] = size(data);
                    X = data(:, 1:n-1);
                    y = data(:, n);
                    found = true;
                    break;
                end
            end
            if ~found
                error(['Unsupported .mat structure. Provide either (X,y) or a matrix (data) ', ...
                    'where last column is label.']);
            end
        end

    otherwise
        error('Unsupported file type. Use .txt or .mat');
end

% Ensure y is a column vector
y = y(:);

%% -------------------- Label conversion: {1,0} -> {1,-1} --------------------
y(y == 0) = -1;

% Basic sanity check for binary classification
classes = unique(y(:));
if numel(classes) ~= 2 || ~all(ismember(classes, [-1, 1]))
    error('This demo expects binary labels in {+1, -1} (or {1,0} which will be converted).');
end

%% -------------------- Stratified Holdout split (70/30) --------------------
test_ratio = 0.25;

idx_pos = find(y == 1);
idx_neg = find(y == -1);

idx_pos = idx_pos(randperm(numel(idx_pos)));
idx_neg = idx_neg(randperm(numel(idx_neg)));

n_pos_test = max(1, round(test_ratio * numel(idx_pos)));
n_neg_test = max(1, round(test_ratio * numel(idx_neg)));

test_idx  = [idx_pos(1:n_pos_test); idx_neg(1:n_neg_test)];
train_idx = setdiff((1:size(X,1))', test_idx);

X_train = X(train_idx, :);
y_train = y(train_idx, :);

X_test  = X(test_idx, :);
y_test  = y(test_idx, :);

%% -------------------- Normalize using training statistics --------------------
mu  = mean(X_train, 1);
sig = std(X_train, 0, 1);
sig(sig == 0) = 1; % numerical safety

X_train = (X_train - mu) ./ sig;
X_test  = (X_test  - mu) ./ sig;

%% -------------------- Fixed hyperparameters (single run) --------------------
C     = 0.000001;     % structural regularization
c     = 1;        % loss regularization
sigma = 100;     % RBF width
a     = 4.7;     % RoBoSS loss parameter
b     = 0.5;     % RoBoSS loss parameter

%% -------------------- Train + Test --------------------
[uu1, uu2, bb1, bb2, acc, train_time] = RoBoSS_TSVM_function( ...
    X_train, y_train, X_test, y_test, a, b, C, c, sigma);

fprintf('\n=== RoBoSS-TSVM ===\n');
fprintf('Dataset: %s\n', data_path);
fprintf('Train size: %d | Test size: %d\n', size(X_train,1), size(X_test,1));
fprintf('Hyperparams: C=%.4g, c=%.4g, sigma=%.4g, a=%.4g, b=%.4g\n', C, c, sigma, a, b);
fprintf('Test accuracy : %.4f %%\n\n', 100*acc);
