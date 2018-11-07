load('D:\Lqd_CX\ML_Shan\data\haberman.mat');
target=data(:,4);
target=target';
data=data(:,1:3);

train_target=target(:,1:245);
test_target=target(:,246:end);

tic;
[PCA_data,selected_vector,selected_lambda]=PCA_by_lqd(data,2);
train_data=PCA_data(1:245,:);
test_data=PCA_data(246:end,:);

Num=10;
Smooth=0.01;
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
[HL1,~,~,~,~,~,Pre_Labels1]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
toc;
%% matlab调用的pca函数
tic;
[COEFF SCORE latent]=pca(data);
PCA_DATA=SCORE(:,1:2);

pca_train_data=PCA_DATA(1:245,:);
pca_test_data=PCA_DATA(246:end,:);

[Prior,PriorN,Cond,CondN]=MLKNN_train(pca_train_data,train_target,Num,Smooth);
[HL2,~,~,~,~,~,Pre_Labels2]=MLKNN_test(pca_train_data,train_target,pca_test_data,test_target,Num,Prior,PriorN,Cond,CondN);

toc;
