function [ PCA_data,selected_vector,selected_lambda] = PCA_by_lqd( data,lower_dimension )
%UNTITLED 此处显示有关此函数的摘要
% Input:
    % data:the original data and the dimension of which is N*d. N is the number of examples,d is the number of features.
    % low_dimension:the goal dimension after the process of pca
% Output:
    % PCA_data:the data after the process of pca
    % selected_lambda:the first lower_dimension maximum eigenvalues after eigening decomposition of matrix
    % selected_vector:The eigenvectors that correspond to the first lower_dimension maximum eigenvalues
%   此处显示详细说明

[N,d]=size(data);
Mean_matrix=mean(data,1);
Data=data-repmat(Mean_matrix,N,1);
cov_matrix=cov(Data);
[selected_vector,selected_lambda]=eigs(cov_matrix,lower_dimension,'la');
PCA_data=Data*selected_vector;

end

