%%%% Filename : calculate_ARL0_empirical.m%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Input arguments
% dist : parsonnet score distribution
% R_0 : Odds ratio of in-control process
% R_1 : Odds ratio of out-of control process
% h : Pre-defined control limit
% M : Number of simulated run length
% N : Number of iteration at each iteration M
% b0 : intercept from logistic regression
% b1 : coefficient of score from logistic regression
%%%% Output
% ARL0 : estimated average run length
% RLset : a set of average run lengths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ARL0, RLset] = calculate_ARL0_empirical(dist,R_0, R_1, h, M,...
    N,b0,b1)

%Run length set
    RL = [];
    e_dist = dist;
% parsonnet score follows the empirical distribution %
    x_t = randsample(e_dist,M,true);

% p_t is a probability obtained from two parameters of logistic regression
% and generated parsonnet score.
    p_t = 0;

for i = 1:M
    c_n = 0;
    p_t = exp(b0+b1*x_t(i))/(1+exp(b0+b1*x_t(i)));
    for j = 1:N
      rng('shuffle');
      y_t = binornd(1,p_t);
      c_n = max(0,c_n+y_t*log(((1-p_t+R_0*p_t)*R_1)/((1-p_t+R_1*p_t)*R_0)) + ...
            (1-y_t)*log(((1-p_t+R_0*p_t))/((1-p_t+R_1*p_t))));  
        if c_n > h
            RL(i) = j;
            break;
        elseif j == N
            RL(i) = N;
        end
     end
end
RLset = RL;
ARL0 = mean(RL);
end