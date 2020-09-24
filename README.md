# Risk-adjusted cumulative sum chart

This session aims to understand the idea of risk-adjusted CUSUM chart in monitoring the quality of care provider.
The quality is defined as any kind of outcome in the particular system. For example, in a cardiac surgery, outcomes are the result of the surgery like alive/died of patients.  

It is natural that patients with their own higher preoperative risks are more likely to die, which produces adverse outcomes in the healthcare system. In this case, even though 
surgeon performs well without any mistake, the patients are more likely to die. This implies that we should adjust preoperative risks of patients to fairly assess the quality of care provider.

The Matlab code here provides how to adjust the preoperative risk (by logistic regression), set the control limit (i.e., acceptable level of performance, Phase I analysis) and thus monitor
test statistics via likelihood ratio test in CUSUM chart. 

Risk-adjusted CUSUM chart to monitor deterioration in the performance of surgeon (i.e., increase in odds ratio, odds in alternative > 1) 
![CUSUM1](https://user-images.githubusercontent.com/69023373/89116141-5e254c00-d456-11ea-8308-c5968e5e65bb.png)

Risk-adjusted CUSUM chart to monitor improvement in the performance of surgeon (i.e., decrease in odds ratio, odds in alternative < 1) 
![CUSUM2](https://user-images.githubusercontent.com/69023373/89116154-7eeda180-d456-11ea-94b7-dffa319bc9d6.png)

# Reference
Steiner, S. H., Cook, R. J., Farewell, V. T., & Treasure, T. (2000). Monitoring surgical performance using risk-adjusted cumulative sum charts. *Biostatistics*, 1(4), 441-452.
