
% load dataset
T = readtable('feature_number_with_class.csv');
% convert feature columns to matrix
data = [T{:,1} T{:,3:12}];

% compute basic statistics
means = mean(data);
vars = var(data);
stdevs = std(data);


%% 2D Scatter Plot

% separate films by class (platform)
platform = data(:,1);
ind1 = platform(:,1) == 1;
netflix = data(ind1,:);
ind2 = platform(:,1) == 2;
hulu = data(ind2,:);
ind3 = platform(:,1) == 3;
prime = data(ind3,:);
% randomize Netflix and Prime
randomizedNetflix = netflix(randperm(size(netflix, 1)),:);
randomizedPrime = prime(randperm(size(prime, 1)),:);
% combine the first 1228 randomized rows (size of Hulu) of each class
randomizedData = vertcat(randomizedNetflix(1:1228,:),hulu,randomizedPrime(1:1228,:));
randomizedPlatform = randomizedData(:,1);
% create labels
netflixLabels = repmat({'Netflix'},length(randomizedPlatform)/3,1);
huluLabels = repmat({'Hulu'},length(randomizedPlatform)/3,1);
primeLabels = repmat({'Prime'},length(randomizedPlatform)/3,1);
platformLabels = vertcat(netflixLabels,huluLabels,primeLabels);

features = {'Year','Age','IMDb Score','Rotten Tomatoes Score (/100)',...
    'Runtime','Genre','Primary Country','Secondary Country',...
    'Primary Language','Secondary Language'};
colors = lines(3);

% plot figure
figure;
hold on
gplotmatrix(randomizedData(:,2:11),[],platformLabels,colors,'.',12,[],'variable',features);
title({'Movie Features','Scatter Plot Matrix'},'FontSize',16)
hold off


%% Preprocessing: Center and scale

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


%% SVD

% 
% X is the original dataset 
% Ur will be the transformed dataset  
% S is covariance matrix (not normalized) 
%
labels = data(:,1);
[U S V] = svd(X,0);
U(:,1) = labels;
% randomize U and attach the region label
U = U(randperm(size(U, 1)),:);
randomizedULabels = U(:,1);
U(:,1) = randomizedULabels;

% makes the randomize U become Ur and then label as r
Ur = U*S;
Ur(:,1) = randomizedULabels;


%% Scree Plots
% 
% Obtain the necessary information for Scree Plots 
% Obtain S^2 (and can also use to normalize S)   
% 
S2 = S^2; 
weights2 = zeros(nfeatures,1);
sumS2 = sum(sum(S2)); 
weightsum2 = 0; 

for i=1:nfeatures 
    weights2(i) = S2(i,i)/sumS2; 
    weightsum2 = weightsum2 + weights2(i); 
    weight_c2(i) = weightsum2; 
end

% plot
figure;
hold on
plot(weights2,'x:b'); 
plot(weight_c2,'x:r'); 
grid; 
title('Scree Plot'); 
xlabel('Feature Index');
ylabel('Cumulative Contribution');
hold off


%% Loading Vectors

for i=1:nfeatures 
    for j=1:nfeatures 
        Vsquare(i,j) = V(i,j)^2; 
        if V(i,j)<0 
            Vsquare(i,j) = Vsquare(i,j)*-1; 
        else  
            Vsquare(i,j) = Vsquare(i,j)*1; 
        end 
    end 
end

% plot
figure;
tiledlayout(4,3);
for i=1:nfeatures
    nexttile;
    bar(Vsquare(:,i),0.5);
    grid;
    ymin = min(Vsquare(:,i)) + (min(Vsquare(:,i))/10);
    ymax = max(Vsquare(:,i)) + (max(Vsquare(:,i))/10);
    axis([0 12 ymin ymax]);
    xlabel('Feature index');
    ylabel('Importance of feature');
    [chart_title, ERRMSG] = sprintf('Loading Vector %d',i);
    title(chart_title);
end


%% 2D Plots - Transformed Space

% separate films by class (platform)
platform = Ur(:,1);
ind1 = platform(:,1) == 1;
netflix = Ur(ind1,:);
ind2 = platform(:,1) == 2;
hulu = Ur(ind2,:);
ind3 = platform(:,1) == 3;
prime = Ur(ind3,:);
% randomize Netflix and Prime
randomizedNetflix = netflix(randperm(size(netflix, 1)),:);
randomizedPrime = prime(randperm(size(prime, 1)),:);
% combine the first 1228 randomized rows (size of Hulu) of each class
randomizedData = vertcat(randomizedNetflix(1:1228,:),hulu,randomizedPrime(1:1228,:));
randomizedPlatform = randomizedData(:,1);
% create labels
netflixLabels = repmat({'Netflix'},length(randomizedPlatform)/3,1);
huluLabels = repmat({'Hulu'},length(randomizedPlatform)/3,1);
primeLabels = repmat({'Prime'},length(randomizedPlatform)/3,1);
platformLabels = vertcat(netflixLabels,huluLabels,primeLabels);

features = {'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'};
colors = lines(3);

% plot figure
figure;
hold on
gplotmatrix(randomizedData(:,2:11),[],platformLabels,colors,'.',12,[],'variable',features);
title({'Movie Features in the Transformed Space (Ur)','Scatter Plot Matrix'},'FontSize',16)
hold off


%% 3D Scatter Plots

% separate films by class (platform)
platform = randomizedData(:,1);
ind1 = platform(:,1) == 1;
netflix3D = randomizedData(ind1,:);
ind2 = platform(:,1) == 2;
hulu3D = randomizedData(ind2,:);
ind3 = platform(:,1) == 3;
prime3D = randomizedData(ind3,:);

figure;
scatter3(netflix3D(:,7),netflix3D(:,8), netflix3D(:,10),'.');
hold on;
scatter3(hulu3D(:,7),hulu3D(:,8), hulu3D(:,10), '.');
scatter3(prime3D(:,7),prime3D(:,8), prime3D(:,10), '.');;
title("PC6 PC7 PC9")
xlabel("PC6")
ylabel("PC7")
zlabel("PC9")
legend([{'Netflix'},{'Hulu'},{'Prime'}])
hold off;

figure;
scatter3(netflix3D(:,7),netflix3D(:,9), netflix3D(:,11),'.');
hold on;
scatter3(hulu3D(:,7),hulu3D(:,9), hulu3D(:,11),'.');
scatter3(prime3D(:,7),prime3D(:,9), prime3D(:,11),'.');
legend([{'Netflix'},{'Hulu'},{'Prime'}])
title("PC6 PC8 PC10")
xlabel("PC6")
ylabel("PC8")
zlabel("PC10")
hold off;

figure;
scatter3(netflix3D(:,3),netflix3D(:,4), netflix3D(:,5),'.');
hold on;
scatter3(hulu3D(:,3),hulu3D(:,4), hulu3D(:,5),'.');
scatter3(prime3D(:,3),prime3D(:,4), prime3D(:,5),'.');
legend([{'Netflix'},{'Hulu'},{'Prime'}])
title("PC2 PC3 PC4")
xlabel("PC2")
ylabel("PC3")
zlabel("PC4")
hold off;

figure;
scatter3(netflix3D(:,2),netflix3D(:,6), netflix3D(:,7), '.');
hold on;
scatter3(hulu3D(:,2),hulu3D(:,6), hulu3D(:,7), '.');
scatter3(prime3D(:,2),prime3D(:,6), prime3D(:,7), '.');
legend([{'Netflix'},{'Hulu'},{'Prime'}])
title("PC1 PC5 PC6")
xlabel("PC1")
ylabel("PC5")
zlabel("PC6")
hold off;

