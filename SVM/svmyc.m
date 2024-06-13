tic;
clear;
clc;
%% import data
load X_tichu60.mat;
X1 = X_tichu60;
X = X1(:,1:1845);
y = X1(:,1846);

c_full = cell(1,length(1000));
c_lle = cell(1,length(1000));
c_lle_corr = cell(1,length(1000));
c_lle_cos = cell(1,length(1000));
c_lle_city = cell(1,length(1000));
c_lle_che = cell(1,length(1000));

y_full = [];
y_lle = [];
y_lle_corr = [];
y_lle_cos = [];
y_lle_city = [];
y_lle_che = [];

n = [];
best = 0;

steps = 1000;
hwait = waitbar(0,'Please Wait>>>>>>>>');
step=steps/100;

for i = 1:1000
    PerStr=fix(i/step);
    str=['Completed ',num2str(PerStr),'%'];
    waitbar(i/steps,hwait,str);
    pause(0.05);
    
    %% Sample set division
    x=size(X,1);
    n1 = randperm(x);
    n = [n;n1];
    q = round(x*0.7);
    %% full
    
    train = X(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    
    test = X(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    
    % %
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_full{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_full = [y_full,PredictLabel];
    %% lle dimension reduction
    Y1 = lle(X',43,21);
    Y1 = Y1';
    
    
    train = Y1(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    test = Y1(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    % %
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_lle{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_lle = [y_lle,PredictLabel];
    %% LLE_corr
    Y2 = lle_corrcoef(X',57,28);
    Y2 = Y2';
    
    
    train = Y2(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    test = Y2(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_lle_corr{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_lle_corr = [y_lle_corr,PredictLabel];
    %% LLE_cos
    Y3 = lle_cos(X',57,27);
    Y3 = Y3';
    
    train = Y3(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    test = Y3(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    % %
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_lle_cos{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_lle_cos = [y_lle_cos,PredictLabel];
    
    
    if test_R2 > best
        best = test_R2;
        besti = i;
    end
    
    %% LLE_city
    Y4 = lle_city(X',56,24);
    Y4 = Y4';
    
    train = Y4(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    test = Y4(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    % %
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_lle_city{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_lle_city = [y_lle_city,PredictLabel];
    %% LLE_che
    Y5 = lle_che(X',47,17);
    Y5 = Y5';
    
    train = Y5(n1(1:q),:);
    y_train = y(n1(1:q),:);
    
    test = Y5(n1(q+1:end),:);
    y_test = y(n1(q+1:end),:);
    
    %%Antlion algorithm for optimization
    [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=ALOSVMcgforclass_cv(10,50,[200,0.00006],[400,0.0005],2,@objfun_svm_cv,y_train,train);
    % %
    bestc = Elite_antlion_position(1);
    bestg = Elite_antlion_position(2);
    cmd = [' -s 3',' -c ',num2str(bestc),' -g ',num2str(bestg)];
    model=svmtrain(y_train,train,cmd);
    
    [ptrain,accuracy1,prob_estimates] = svmpredict(y_train,train,model);
    train_mse = accuracy1(2);
    train_RMSE = sqrt(train_mse);
    train_R2 = accuracy1(3);
    train_mae = mean(abs(y_train - ptrain));
    
    [PredictLabel,accuracy2,prob_estimates] = svmpredict(y_test,test,model);
    test_mse = accuracy2(2);
    test_RMSE = sqrt(test_mse);
    test_R2 = accuracy2(3);
    test_mae = mean(abs(y_test - PredictLabel));
    c_lle_che{i} = {train_RMSE,train_R2,test_RMSE,test_R2,train_mae,test_mae};
    y_lle_che = [y_lle_che,PredictLabel];
end
toc;

full_train_RMSE = [];
full_train_R2 = [];
full_test_RMSE = [];
full_test_R2 = [];

lle_train_RMSE = [];
lle_train_R2 = [];
lle_test_RMSE = [];
lle_test_R2 = [];

lle_corr_train_RMSE = [];
lle_corr_train_R2 = [];
lle_corr_test_RMSE = [];
lle_corr_test_R2 = [];

lle_cos_train_RMSE = [];
lle_cos_train_R2 = [];
lle_cos_test_RMSE = [];

lle_city_train_RMSE = [];
lle_city_train_R2 = [];
lle_city_test_RMSE = [];
lle_city_test_R2 = [];

lle_che_train_RMSE = [];
lle_che_train_R2 = [];
lle_che_test_RMSE = [];
lle_che_test_R2 = [];

for i = 1 : 1000
    full_train_RMSE = [full_train_RMSE,c_full{1,i}(1,1)];
    full_train_R2 = [full_train_R2,c_full{1,i}(1,2)];
    full_test_RMSE = [full_test_RMSE,c_full{1,i}(1,3)];
    full_test_R2 = [full_test_R2,c_full{1,i}(1,4)];
    
    lle_train_RMSE = [lle_train_RMSE,c_lle{1,i}(1,1)];
    lle_train_R2 = [lle_train_R2,c_lle{1,i}(1,2)];
    lle_test_RMSE = [lle_test_RMSE,c_lle{1,i}(1,3)];
    lle_test_R2 = [lle_test_R2,c_lle{1,i}(1,4)];
    
    lle_corr_train_RMSE = [lle_corr_train_RMSE,c_lle_corr{1,i}(1,1)];
    lle_corr_train_R2 = [lle_corr_train_R2,c_lle_corr{1,i}(1,2)];
    lle_corr_test_RMSE = [lle_corr_test_RMSE,c_lle_corr{1,i}(1,3)];
    lle_corr_test_R2 = [lle_corr_test_R2,c_lle_corr{1,i}(1,4)];
    
    lle_cos_train_RMSE = [lle_cos_train_RMSE,c_lle_cos{1,i}(1,1)];
    lle_cos_train_R2 = [lle_cos_train_R2,c_lle_cos{1,i}(1,2)];
    lle_cos_test_RMSE = [lle_cos_test_RMSE,c_lle_cos{1,i}(1,3)];
    lle_cos_test_R2 = [lle_cos_test_R2,c_lle_cos{1,i}(1,4)];
    
    lle_city_train_RMSE = [lle_city_train_RMSE,c_lle_city{1,i}(1,1)];
    lle_city_train_R2 = [lle_city_train_R2,c_lle_city{1,i}(1,2)];
    lle_city_test_RMSE = [lle_city_test_RMSE,c_lle_city{1,i}(1,3)];
    lle_city_test_R2 = [lle_city_test_R2,c_lle_city{1,i}(1,4)];
    
    lle_che_train_RMSE = [lle_che_train_RMSE,c_lle_che{1,i}(1,1)];
    lle_che_train_R2 = [lle_che_train_R2,c_lle_che{1,i}(1,2)];
    lle_che_test_RMSE = [lle_che_test_RMSE,c_lle_che{1,i}(1,3)];
    lle_che_test_R2 = [lle_che_test_R2,c_lle_che{1,i}(1,4)];
end
full_train_RMSE1 = mean(cell2mat(full_train_RMSE));
full_train_R21 = mean(cell2mat(full_train_R2));
full_test_RMSE1 = mean(cell2mat(full_test_RMSE));
full_test_R21 = mean(cell2mat(full_test_R2));

lle_train_RMSE1 = mean(cell2mat(lle_train_RMSE));
lle_train_R21 = mean(cell2mat(lle_train_R2));
lle_test_RMSE1 = mean(cell2mat(lle_test_RMSE));
lle_test_R21 = mean(cell2mat(lle_test_R2));

lle_corr_train_RMSE1 = mean(cell2mat(lle_corr_train_RMSE));
lle_corr_train_R21 = mean(cell2mat(lle_corr_train_R2));
lle_corr_test_RMSE1 = mean(cell2mat(lle_corr_test_RMSE));
lle_corr_test_R21 = mean(cell2mat(lle_corr_test_R2));

lle_cos_train_RMSE1 = mean(cell2mat(lle_cos_train_RMSE));
lle_cos_train_R21 = mean(cell2mat(lle_cos_train_R2));
lle_cos_test_RMSE1 = mean(cell2mat(lle_cos_test_RMSE));
lle_cos_test_R21 = mean(cell2mat(lle_cos_test_R2));

lle_city_train_RMSE1 = mean(cell2mat(lle_city_train_RMSE));
lle_city_train_R21 = mean(cell2mat(lle_city_train_R2));
lle_city_test_RMSE1 = mean(cell2mat(lle_city_test_RMSE));
lle_city_test_R21 = mean(cell2mat(lle_city_test_R2));

lle_che_train_RMSE1 = mean(cell2mat(lle_che_train_RMSE));
lle_che_train_R21 = mean(cell2mat(lle_che_train_R2));
lle_che_test_RMSE1 = mean(cell2mat(lle_che_test_RMSE));
lle_che_test_R21 = mean(cell2mat(lle_che_test_R2));
%%

[a,b] = sort(cell2mat(lle_cos_test_R2));
i_100 = b(:,901:1000);
k = cell2mat(lle_cos_test_RMSE(:,i_100));

full_test_RMSE100 = cell2mat(full_test_RMSE(:,i_100));
full_test_R2100 = cell2mat(full_test_R2(:,i_100));
full_test_RMSE1 = mean(full_test_RMSE100);
full_test_R21 = mean(full_test_R2100);

lle_test_RMSE100 = cell2mat(lle_test_RMSE(:,i_100));
lle_test_R2100 = cell2mat(lle_test_R2(:,i_100));
lle_test_RMSE1 = mean(lle_test_RMSE100);
lle_test_R21 = mean(lle_test_R2100);

lle_corr_test_RMSE100 = cell2mat(lle_corr_test_RMSE(:,i_100));
lle_corr_test_R2100 = cell2mat(lle_corr_test_R2(:,i_100));
lle_corr_test_RMSE1 = mean(lle_corr_test_RMSE100);
lle_corr_test_R21 = mean(lle_corr_test_R2100);

lle_cos_test_RMSE100 = cell2mat(lle_cos_test_RMSE(:,i_100));
lle_cos_test_R2100 = cell2mat(lle_cos_test_R2(:,i_100));
lle_cos_test_RMSE1 = mean(lle_cos_test_RMSE100);
lle_cos_test_R21 = mean(lle_cos_test_R2100);

lle_city_test_RMSE100 = cell2mat(lle_city_test_RMSE(:,i_100));
lle_city_test_R2100 = cell2mat(lle_city_test_R2(:,i_100));
lle_city_test_RMSE1 = mean(lle_city_test_RMSE100);
lle_city_test_R21 = mean(lle_city_test_R2100);

lle_che_test_RMSE100 = cell2mat(lle_che_test_RMSE(:,i_100));
lle_che_test_R2100 = cell2mat(lle_che_test_R2(:,i_100));
lle_che_test_RMSE1 = mean(lle_che_test_RMSE100);
lle_che_test_R21 = mean(lle_che_test_R2100);
%% box plot

figure

h = boxplot([lle_city_test_R2100',lle_corr_test_R2100',full_test_R2100',lle_test_R2100',lle_che_test_R2100',lle_cos_test_R2100'],'Labels',{'LLE_man-SVM','LLE_cor-SVM','Full-SVM','LLE-SVM','LLE_che-SVM','LLE_cos-SVM'},'colors','k');


set(h,'LineWidth',2);
box off
xlabel('Model','Fontname', 'Times New Roman')
ylabel('R^2','Fontname', 'Times New Roman')
set(gca, 'FontSize',25)

figure

h = boxplot([lle_city_test_RMSE100',lle_corr_test_RMSE100',full_test_RMSE100',lle_test_RMSE100',lle_che_test_RMSE100',lle_cos_test_RMSE100'],'Labels',{'LLE_man-SVM','LLE_cor-SVM','Full-SVM','LLE-SVM','LLE_che-SVM','LLE_cos-SVM'},'colors','k');

set(h,'LineWidth',2);
box off
xlabel('Model','Fontname', 'Times New Roman')
ylabel('RMSE','Fontname', 'Times New Roman')
set(gca, 'FontSize',25)

toc;
