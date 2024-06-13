%% SVM_Objective Function SVM_目标函数
function f=objfun_svm_cv(cg,train_labels,train)

cmd = [' -s num2str(3)',' -v ',num2str(10),' -c ',num2str(cg(1)),' -g ',num2str(cg(2))];
model = svmtrain(train_labels,train,cmd);

f=1-model/100;
