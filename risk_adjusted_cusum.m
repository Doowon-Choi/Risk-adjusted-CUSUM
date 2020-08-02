% import data set using dataset array function. 
% we can use xlsread without header
clear;
clc;
format compact;

data = dataset('XLSFile','SGH_data.xls');
var_name = data.Properties.VarNames(:);
[row, col] = size(data);

% descriptive data analysis
s = summary(data);
s.Variables.Size;
max_stats = max(double(data(:,end-1)),2);


% find date 1993 12 29 (the last operation date for the first 2 years
% to extract subset data for logistic regression
index = 0;
for i = 1:row
    if char(data.OP_DATE(i))=='1993-12-29';
       index = i;
       break
    end
end

%%%%%% estimate parameters of Logistic regression %%%%%
%%% select subset data for Logistic regression %%%
data_logistic = data(1:index, {'in_out','Parsonnet'});
data_logistic = set(data_logistic, 'VarNames', {'y', 'x'});

%replace response y of 2 with 1 for binary logistic regression
for ii = 1:index
    if data_logistic.y(ii)== 2;
        data_logistic.y(ii) = 0;
    end
end

histogram(data_logistic.x);
title('Distribution of parsonnet score');

%  pdf = fitdist(data_logistic.x, 'exponential');
%  mu = expfit(data_logistic.x)
%  syntheticdata = exprnd(9,1000,1);
%  qqplot(data_logistic.x,syntheticdata)
%
pdf1 = histfit(data_logistic.x,8.9, 'exponential')
% display(pdf)
% [f,x] = ecdf(data_logistic

modelspec = 'y ~ x';
% the coefficienst are different from that of the paper, but are the same as the
% result of logistic regression perfromed in R 
model = fitglm(data_logistic, modelspec, 'Distribution', 'binomial');
display(model)

% parameters
parameter = model.Coefficients.Estimate;
b0 = parameter(1); 
b1 = parameter(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate ARL0 by simulation given h and R1(odds ratio of alternative
% hypothesis.Details are included in calculate_ARL0_empirical.m file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R_0 = 1;
R_1 = 2;
h = 4.5;
M = 15000;
N = 20000;
dist = data_logistic.x;

% aa = randsample(data_logistic.x,2200,true);
% histfit(aa,[],'exponential');

% The fllowing function assumes that parsonnet score distribution follows
% uniform distribution
% ARL = calculate_ARL0_uniform(R_0, R_1, h, M, N,b0,b1);


% The fllowing function assumes that parsonnet score distribution follows
% empricial distribution of it. That is, the sampled random numbers are
% generated from the empricial distribution.
tic;[ARL0, RL] = calculate_ARL0_empirical(dist,R_0, R_1, h, M, N,b0,b1); toc;

%save distribution of average run length
histfit(RL,[],'NegativeBinomial');
title('Distribution of average run length');
% print('Distribution of average run length','-dpng');

save risk_adjusted

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate control limit h by simulation given ARL0 and R1(odds ratio 
% of alternative hypothesis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARL 562, h 2.5. These figures are from simulation.
clear h;
dist = data_logistic.x;
A0 = 550;
h_l = 2;
h_u = 3;
e = 15;
K = 10000; % the maximum number of iterations involved in search
M = 10000;
N = 10000;
R_0 = 1;
R_1 = 2;
optimal_h = 0;

for ii = 1 : K
     h = (h_l + h_u)/2;
     
     % ARL0 is calculated based on uniform distribution
     % ARL0 = calculate_ARL0_unifrom(R_0, R_1, h, M, N, b0, b1);
     
     % Random sampling with replacement
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
    fprintf('iteration %d\n',ii);
    
    if optimal_h==0 && ii==K
       optimal_h = h;
       fprintf('Cannot converge and returns control limit at the final step :%.2f\n',optimal_h);
    end
        if mod(ii,100)==0
            fprintf('iteration %d\n',ii);
        end
end    

save risk_adjusted;

%%%%%%%%% Figure 2 %%%%%%%%%%
% need to classify surgeons %
data_valid = data(index+1:end, {'OP_DATE','surgeon','in_out','Parsonnet'});
% data_valid = data(1:index, {'surgeon','in_out','Parsonnet'});
data_valid = set(data_valid, 'VarNames', {'date','surgeon','y', 'x'});

in_control = data(1:index, {'surgeon','in_out','Parsonnet'});
in_control= set(in_control, 'VarNames', {'surgeon','y', 'x'});
low_risk_data = in_control(in_control.x==0,:);
low_risk_data = categorical(low_risk_data.y);
summary(low_risk_data);
[row1, col1] = size(data_valid);

for j = 1:row1
    if data_valid.y(j)== 2;
        data_valid.y(j) = 0;
    end
end

data_valid.x= categorical(data_valid.x);
% data_valid.y= categorical(data_valid.y);

dv_surgeon1 = data_valid(data_valid.surgeon==1,:);
dv_surgeon2 = data_valid(data_valid.surgeon==2,:);
dv_surgeon3 = data_valid(data_valid.surgeon==3,:);
dv_surgeon5 = data_valid(data_valid.surgeon==5,:);
dv_surgeon6 = data_valid(data_valid.surgeon==6,:);
dv_surgeon7 = data_valid(data_valid.surgeon==7,:);
dv_surgeon4 = data_valid(data_valid.surgeon==4,:);

summary(dv_surgeon7.y)
save ra_validation;
load ra_validation;
c_n = 0;
cn_set=[];
cn_set(1)=0;
wt_set=[];
data_valid_1 = dv_surgeon2;
[row2, col2] = size(data_valid_1);
R_1 = 0.5;
b0=-3.68;
b1=0.077;
% focus only one surgeon 2
for i = 1:row2
    x_t=data_valid_1.x(i);
    p_t = exp(b0+b1*x_t)/(1+exp(b0+b1*x_t));
    y_t = data_valid_1.y(i);
    w_t = log((((1-p_t+R_0*p_t)*R_1/(1-p_t+R_1*p_t)*R_0))^y_t...
        *((1-p_t+R_0*p_t)/(1-p_t+R_1*p_t))^(1-y_t));
    wt_set(i) = w_t;
    c_n = min(0,c_n - w_t);
    cn_set(i+1) = c_n;
end

x =1:length(cn_set);
plot(x,cn_set)
ylabel('CUSUM Zt')
xlabel('patient number')
axis([0 350 -4 0]);
title('Distribution of parsonnet score');
line('XData', [0 350], 'YData', [-3.35 -3.35], 'LineStyle', ':', ...
    'LineWidth', 0.6);

print('surgeon2','-dpng');

% ra cusum by one specific  %
% generate trainee surgeons 567 except 4 5 6 7 %
c_n = 0;
cn_set=[];
cn_set(1)=0;
wt_set=[];
R_1=2;
for i = 1:row1
    if (data_valid.surgeon(i)~=5 && data_valid.x(i)>=2) || ...
      (data_valid.surgeon(i)~=6 && data_valid.x(i)>=2) || ...
      (data_valid.surgeon(i)~=7 && data_valid.x(i)>=2) || ...
       (data_valid.surgeon(i)~=3 && data_valid.x(i)>=2) 
%         x_t= poissrnd(0,1); % exprnd(5.1,1)
%         p_t = exp(b0+b1*x_t)/(1+exp(b0+b1*x_t));
%         y_t = binornd(1,p_t);
%         w_t = log((((1-p_t+R_0*p_t)*R_1/(1-p_t+R_1*p_t)*R_0))^y_t...
%           *((1-p_t+R_0*p_t)/(1-p_t+R_1*p_t))^(1-y_t));
%         wt_set(i) = w_t;
%         c_n = max(0,c_n+w_t); 
%         cn_set(i+1) = c_n;
      w_t=0;
      wt_set(i) = w_t;
      c_n = max(0,c_n+w_t); 
      cn_set(i+1) = c_n;
 else
        x_t=data_valid.x(i);
        p_t = exp(b0+b1*x_t)/(1+exp(b0+b1*x_t));
        y_t = data_valid.y(i);
        w_t = log((((1-p_t+R_0*p_t)*R_1/(1-p_t+R_1*p_t)*R_0))^y_t...
          *((1-p_t+R_0*p_t)/(1-p_t+R_1*p_t))^(1-y_t));
        wt_set(i) = w_t;
        c_n = max(0,c_n+w_t); 
        cn_set(i+1) = c_n;
  end
end
x =1:length(cn_set);
plot(x/1000,cn_set);
ylabel('CUSUM Xt')
xlabel('years from start of 1994')
axis([0 5 0 6]);
line('XData', [0 5], 'YData', [4.5 4.5], 'LineStyle', ':', ...
    'LineWidth', 0.6);

print('trainees','-dpng');
out_data = data_valid(:,3:4);
export(out_data,'XLSFile','data_valid.xlsx');

% stnadard cusum %
% generate trainee surgeons 1 2 3 except 4 5 6 7 %
c_n = 0;
cn_set=[];
cn_set(1)=0;
wt_set=[];
R_1=2;
for i = 1:row1
 if  data_valid.surgeon(i)~=3 && data_valid.surgeon(i)~=5 ...
%       &&  data_valid.surgeon(i)~=6
%         w_t=0;
%         wt_set(i) = w_t;
%         c_n = max(0,c_n+w_t); 
        cn_set(i+1) = c_n;
 else
        x_t=data_valid.x(i);
        p_t = exp(b0+b1*x_t)/(1+exp(b0+b1*x_t));
        y_t = data_valid.y(i);
        w_t = log((((1-p_t+R_0*p_t)*R_1/(1-p_t+R_1*p_t)*R_0))^y_t...
          *((1-p_t+R_0*p_t)/(1-p_t+R_1*p_t))^(1-y_t));
        wt_set(i) = w_t;
        c_n = max(0,c_n+w_t); 
        cn_set(i+1) = c_n;
  end
end
x =1:length(cn_set);
plot(x/1000,cn_set);
ylabel('CUSUM Zt')
xlabel('years from start of 1994')
axis([0 5 0 6]);
line('XData', [0 5], 'YData', [4.5 4.5], 'LineStyle', ':', ...
    'LineWidth', 0.6);
% axis([0 5 -6 0]);
% line('XData', [0 5], 'YData', [-4 -4], 'LineStyle', ':', ...
%     'LineWidth', 0.6);

[m, m_index] = max(wt_set);
wt_set(112);
cn_set(112);
x =1:length(cn_set);
plot(x,cn_set);
plot(x,wt_set);
%% Using estimate_control_limit_cusum m file
% ARL 421, h 2.5
clear h;
A0 = 600;
h_l = 1.95;
h_u = 2.08;
e = 30;
K = 1000; 
M =15000;
N = 5000;
R_0 = 1;
R_1 = 2;
initial_dist = data_logistic.x;
summary(data_logistic)
index_lower = find(initial_dist < median(initial_dist));
lower_dist = initial_dist(index_lower);

index_upper= find(initial_dist > median(initial_dist));
upper_dist = initial_dist(index_upper);

%vary dist : uniform, exponential, empirical and investigate control limit#
% dist = data_logistic.x;
 dist1 = exprnd(2,100000,1);
 dist2 = exprnd(4,100000,1);
 dist3 = exprnd(6,100000,1);
 dist4 = exprnd(8,100000,1);
 dist5 = exprnd(10,100000,1);
 dist6 = exprnd(12,100000,1);
 dist7 = exprnd(14,100000,1);
 % dist = 69.*rand(100000,1);
 dist_set = [dist1, dist2, dist3, dist4, dist5, dist6, dist7];
 h=[];
 for k = 1: 7
%      dist = dist_set(:,k);
dist = dist1;
 h = estimate_control_limit_cusum(dist,A0, h_l, h_u,...
     e, K,M,N,R_0,R_1,b0,b1);
%      h(k) = estimate_control_limit_cusum(dist,A0, h_l, h_u,...
%     e, K,M,N,R_0,R_1,b0,b1);
fprintf('iteration  : %.2f %d\n',k);
 end
 A
 
dist_low = lower_dist;
dist_up = upper_dist;
% dist =exprnd(4,100000,1); 
h = 2.4;
tic;ARL_lower= calculate_ARL0_empirical(dist_low,R_0, R_1, h, M, N, b0, b1);toc;


ARL_upper= calculate_ARL0_empirical(dist_up,R_0, R_1, h, M, N, b0, b1);
ARL_upper

% ARL by exponential %
h=2;
dist_exp = exprnd(6,100000,1);
ARL_exp= calculate_ARL0_empirical(dist_exp,R_0, R_1, h, M, N, b0, b1);
ARL_exp
h_set = [1.96, 1.98, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45];
result_tot = [];
for t = 1:7
    a = 2*t + 6;
    dist_e = exprnd(a,100000,1);
    result_sub=[];
    for k = 1:12
    h = h_set(k);    
    ARL_exp= calculate_ARL0_empirical(dist_e,R_0, R_1, h, M, N, b0, b1);
    result_sub=[result_sub, ARL_exp];
    end
    result_tot=[result_tot; result_sub];
end
    
% plot h versus mean value %
x = (2:2:20);
cl = [1.8, 1.9, 1.95, 2.05, 2.1, 2.15, 2.2, 2.23, 2.24, 2.25];
in_con_arl = [781, 702,624,571, 526, 492, 470, 447, 432, 430]; 
plot(x,cl);
plot(x, in_con_arl);
ylabel('In-control ARL')
xlabel('Average Parsonnet Score')
axis([2 20 350 820]);
line('XData', [0 5], 'YData', [4.5 4.5], 'LineStyle', ':', ...
    'LineWidth', 0.6);
 tic; optimal_h = estimate_control_limit_cusum(dist1,A0, h_l, h_u,...
    e, K,M,N,R_0,R_1,b0,b1); toc;

clear h;
%  tic; h_low = estimate_control_limit_cusum(dist_low,A0, h_l, h_u,...
%     e, K,M,N,R_0,R_1,b0,b1); toc;
% 
%  tic; h_up = estimate_control_limit_cusum(dist_up,A0, h_l, h_u,...
%     e, K,M,N,R_0,R_1,b0,b1); toc;