%% a litte clean work
%close all;
clear;
clc;
format compact;
%% import data

load laohuaYM_noAvg.csv
X = laohuaYM_noAvg(2:316,1:1845);
y = laohuaYM_noAvg(2:316,1846);
nirs_data = X;
content_data=y;

train_data=nirs_data;
train_label=content_data;
%% 
[m_size n_size]=size(train_data); 
k=5000;

residual_L=zeros(k,m_size);
steps=k;
hwait=waitbar(0,'Please Wait>>>>>>>>');
step=steps/100;
for gen=1:k
    	%% progress bar
    if steps-gen<=1
        waitbar(gen/steps,hwait,'ready to complete');
        pause(0.05);
    else
        PerStr=fix(gen/step);
        str=['Completed ',num2str(PerStr),'%'];
        waitbar(gen/steps,hwait,str);
        pause(0.05);
    end
   
    
    [train, test] = crossvalind('HoldOut',m_size,1/3); 
    data_train=train_data(train,:); 
    label_train=train_label(train,:);
    data_test=train_data(test,:); 
    label_test=train_label(test,:);
    %% PLS
    ab_train=[data_train label_train];  
    mu=mean(ab_train);sig=std(ab_train); %计算每一列的标准差。
    rr=corrcoef(ab_train);   %corrcoef(A) 返回 A 的相关系数的矩阵，其中 A 的列表示随机变量，行表示观测值。
    ab=zscore(ab_train); %返回X的每个元素的z分数，使得X的列居中以具有平均值0。Z与X的大小相同。
    a=ab(:,[1:n_size]);b=ab(:,[n_size+1:end]);  
    ncomp=NCOMP_BestNumber_Search_RMSECV(data_train,label_train(:,1),5,15); 
%     
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] =plsregress(a,b,ncomp);
    n=size(a,2); mm=size(b,2);
    beta3(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA([2:end],:).*sig(n+1:end); 
    beta3([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA([2:end],:); 
    ab_test=[data_test label_test];
    a1=ab_test(:,[1:n_size]);b1=ab_test(:,[n_size+1:end]); 
    yhat_test=repmat(beta3(1,:),[size(a1,1),1])+ab_test(:,[1:n])*beta3([2:end],:); 
    temp_residual= abs(yhat_test-label_test); 
    [test_m,test_n]=size(temp_residual);
    num=0;
    j=1;
    for j=1:m_size
        if test(j)==1
            num=num+1;
            residual_L(gen,j)=temp_residual(num,1);
        end
    end
end
close(hwait);
%% calculated the mean and SD of the residuals 
for i=1:m_size
    R_L=residual_L(:,i);
    R_L(R_L==0)=[];
    Mean_Residual_L(i)=mean(R_L);
    Var_Residual_L(i)=var(R_L);
    Number(i)=i;
end
figure(3)
%Draw a two-dimensional scatter plot and number it
plot(Mean_Residual_L,Var_Residual_L,'r.');%Draw a scatter plot
hold on
for i=1:max(size(Mean_Residual_L))
    c = num2str(i);
    text(Mean_Residual_L(i),Var_Residual_L(i),c);
end
xlabel('mean');
ylabel('variance');
hold off