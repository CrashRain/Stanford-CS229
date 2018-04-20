clc;
clear all;

format compact

tau = .5; % Could select from 0.01 to 5.0
resolution = 200; % 50 ~ 200

% load data
[X_train, y_train] = load_data;

% run regression
plot_lwlr(X_train, y_train, tau, resolution);