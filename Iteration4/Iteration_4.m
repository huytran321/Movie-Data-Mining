% load dataset
T = readtable("feature number with class.csv");
% convert feature columns to matrix
data = [T{:,1} T{:,3:12}];

means = mean(data);
vars = var(data);
stdevs = std(data);

% set up variables
[nrows, ncols] = size(data);
X = zeros([nrows,ncols]);

% Mean center data 
for i=1:ncols 
    for j=1:nrows 
        X(j,i) = -(means(:,i) - data(j,i)); 
    end 
end 
mean(X)

% Scale data
for i=1:ncols 
    for j=1:nrows 
        X(j,i) = X(j,i) / stdevs(:,i); 
    end 
end         
var(X)

% X is the original dataset 
% Ur will be the transformed dataset  
% S is covariance matrix (not normalized) 
%
labels = data(:,1);
[U S V] = svd(X,0);
U(:,1) = labels;
%Randomize U and attach the region label
U = U(randperm(size(U, 1)),:);
randomizedULabels = U(:,1);
U(:,1) = randomizedULabels;

%Makes the randomize U become Ur and then label as r
Ur = U*S;
Ur(:,1) = randomizedULabels;
features = {'Streaming Platform','Year','Age','IMDb Score','Rotten Tomatoes Score (/100)',...
    'Runtime','Genre','Primary Country','Secondary Country',...
    'Primary Language','Secondary Language'};

Table_Ur = array2table(Ur,'VariableNames',features);
Table_U = array2table(U,'VariableNames',features);
