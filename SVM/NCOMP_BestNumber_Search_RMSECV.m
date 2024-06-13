function BestNum=NCOMP_BestNumber_Search_RMSECV(train_data,train_label,k,Num)
%The optimal number of principal components was determined based on the correction set RMSECV
%BestNum is the best number of principal components
%Num is the maximum number of principal components to be calculated
%k is the cross-validated k-fold coefficient in k-fold
BestNum=0;
PRESS=Inf;
for gen=1:Num    
    RMSECV_temp=pls_crossvalind(train_data,train_label,k,gen);
    if PRESS>RMSECV_temp
        PRESS=RMSECV_temp;
        BestNum=gen;
    end
end