%%% This file is desigend to find the optimal control limit h
%%% The calculate_ARL0_empirical.m is designed to estimate ARL0 given h
%%% and alternative odds ratio R_1.
%%% In-control : R_0 = 1, out-of-control : R_1

%%%%%%%%% find optimal control limit by bisection method %%%%%%%%%%%
%%% Input argument
%%% dist : parsonnet score distribution
%%% A0 : pre-sepcified ARL0
%%% [h_l, h_u] : interval of h
%%% e = the required estimation accuracy, allowance to deviation from A0
%%% K = the maximum number of iterations in searching control limit h
%%% M = the maximum number of iteration for computing ARL0 given h and  R_1
%%% This index is required for calculate_ARL0.m
%%% N = the maximum number of Run length at each interation M

% ARL 421, h 2.5
% A0 = 420;
% h_l = 2.2;
% h_u = 2.8;
% e = 15;
% K = 20000; 
% M = 10000;
% N = 5000;
% R_0 = 1;
% R_1 = 2;

function optimal_h = estimate_control_limit_cusum...
    (dist, A0, h_l, h_u, e, K,M,N,R_0,R_1,b0,b1) 
 optimal_h = 0;
for ii = 1 : K
    h = (h_l + h_u)/2;
    ARL0 = calculate_ARL0_empirical(dist,R_0, R_1, h, M, N, b0, b1);
    
    if A0-e <= ARL0 && ARL0 <= A0+e
        optimal_h = h;
        break;
    elseif A0 < ARL0
        h_l = h_l;
        h_u = (h_l + h_u)/2;
    elseif A0 > ARL0
        h_l = (h_l + h_u)/2;
        h_u = h_u;
    end
       
    if optimal_h==0 && ii==K
       optimal_h = h;
       fprintf('Cannot converge and returns control limit at the final step :%.2f\n',h);
    end
 end    
    fprintf('estimated control limit  : %.2f given ARL0 %d\n',optimal_h,A0);
end