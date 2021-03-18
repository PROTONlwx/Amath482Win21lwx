%% classify 2 digits
clear all; close all; clc;

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
images = im2double(images);
imgSize = size(images);
images = reshape(images,[imgSize(1) * imgSize(2), imgSize(3)]);
images = bsxfun(@minus, images, mean(images,2));
[U,S,V] = svd(images,'econ');

figure(1)
plot(diag(S),'ko','Linewidth',1)
set(gca,'Fontsize',16)
title('Singular Values')
xlabel('singular values')
ylabel('magnitude')

figure(2)
Y = U' * images;
colors = [[0 0.4470 0.7410];[0.8500 0.3250 0.0980];[0.9290 0.6940 0.1250];[0.4940 0.1840 0.5560];[1 0 0];[0 1 0];[0 0 1];[0 1 1];[1 0 1];[1 1 0]];
for j = 0:9
    idx = find(labels==j);
    plot3(Y(2,idx), Y(3,idx), Y(5,idx), 'o', 'Color', colors(j+1,:), 'displayname', sprintf('NUM: %s',string(j))), hold on
end
set(gca,'Fontsize',16)
legend()
title('Projection of digits image on three modes')
xlabel('Mode 1')
ylabel('Mode 2')
zlabel('Mode 3')

%%
clear all; close all; clc;

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
images = im2double(images);
imgSize = size(images);
images = reshape(images,[imgSize(1) * imgSize(2), imgSize(3)]);
[U,S,V] = svd(images,'econ');

original = reshape(images, imgSize);
appro1 = reshape(U(:,1)*S(1,1)*V(:,1)', imgSize);
appro10 = reshape(U(:,1:10)*S(1:10,1:10)*V(:,1:10)', imgSize);
appro20 = reshape(U(:,1:20)*S(1:20,1:20)*V(:,1:20)', imgSize);
appro30 = reshape(U(:,1:30)*S(1:30,1:30)*V(:,1:30)', imgSize);
appro40 = reshape(U(:,1:40)*S(1:40,1:40)*V(:,1:40)', imgSize);
appro50 = reshape(U(:,1:50)*S(1:50,1:50)*V(:,1:50)', imgSize);
appro60 = reshape(U(:,1:60)*S(1:60,1:60)*V(:,1:60)', imgSize);
appro70 = reshape(U(:,1:70)*S(1:70,1:70)*V(:,1:70)', imgSize);
subplot(3,3,1), imshow(original(:, :, 1));
title('Original Picture')
subplot(3,3,2), imshow(appro1(:, :, 1));
title('Rank 1 approximation')
subplot(3,3,3), imshow(appro10(:, :, 1));
title('Rank 10 approximation')
subplot(3,3,4), imshow(appro20(:, :, 1));
title('Rank 20 approximation')
subplot(3,3,5), imshow(appro30(:, :, 1));
title('Rank 30 approximation')
subplot(3,3,6), imshow(appro40(:, :, 1));
title('Rank 40 approximation')
subplot(3,3,7), imshow(appro50(:, :, 1));
title('Rank 50 approximation')
subplot(3,3,8), imshow(appro60(:, :, 1));
title('Rank 60 approximation')
subplot(3,3,9), imshow(appro70(:, :, 1));
title('Rank 70 approximation')
%%
clear all; close all; clc;

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
images = im2double(images);
imgSize = size(images);
images = reshape(images,[imgSize(1) * imgSize(2), imgSize(3)]);
[U,S,V] = svd(images,'econ');
feature = 50;
Y = U' * images;
accuracy = [];
for num1 = 0:9
    for num2 = num1 + 1 : 9
        [threshold,w,sortdog,sortcat] = digits_trainer_2(num1, num2, feature, Y, labels);
        accuracy = [accuracy; [num1 num2 checkAccuracy(num1, num2, feature, U, threshold, w) checkTrainAccuracy(num1, num2, feature, U, threshold, w)]];
    end
end
hardest = accuracy(accuracy(:,3) == min(accuracy(:,3)),:); % [4,9,0.947262682069312]
easiest = accuracy(accuracy(:,3) == max(accuracy(:,3)),:); % [0,4,0.998470948012233]

%% classify three digits
clear all; close all; clc;

accuracy = [];
for num1 = 0:9
    for num2 = num1 + 1 : 9
        for num3 = num2 + 1 : 9
            accuracy = [accuracy; [num1 num2 num3 classify_3(num1, num2, num3)]];
        end
    end
end
hardest = accuracy(accuracy(:,4) == min(accuracy(:,4)),:); % [3,5,8,0.890820584144645]
easiest = accuracy(accuracy(:,4) == max(accuracy(:,4)),:); % [0,1,4,0.996771068776235]

%% using decision tree
clear all; close all; clc;

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
images = im2double(images);
imgSize = size(images);
images = reshape(images,[imgSize(1) * imgSize(2), imgSize(3)]);
[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImages = im2double(testImages);
testImagesSize = size(testImages);
testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

tree=fitctree(images',labels);
test_labels = predict(tree,testImages');
train_labels = predict(tree,images');
train_accuracy = sum(labels==train_labels)/length(train_labels);
test_accuracy = sum(testLabels==test_labels)/length(test_labels);
view(tree,'Mode','graph');
save('DT.mat','tree');
%% using SVM 
clear all; close all; clc;

[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
images = im2double(images);
imgSize = size(images);
images = reshape(images,[imgSize(1) * imgSize(2), imgSize(3)]);

[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImages = im2double(testImages);
testImagesSize = size(testImages);
testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);
    
Mdl = fitcecoc(images',labels);

train_labels = predict(Mdl, images');
test_labels = predict(Mdl,testImages');

train_accuracy = sum(labels==train_labels)/length(train_labels);
test_accuracy = sum(testLabels==test_labels)/length(test_labels);
save('SVM.mat','Mdl');
%% continued
load('SVM.mat')
load('DT.mat')
[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
train_images = im2double(train_images);
imgSize = size(train_images);
train_images = reshape(train_images,[imgSize(1) * imgSize(2), imgSize(3)]);

[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImages = im2double(testImages);
testImagesSize = size(testImages);
testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

SVMlabels = predict(Mdl,testImages');
treeLabels = predict(tree,testImages');
confusionchart(testLabels,SVMlabels,'RowSummary','total-normalized');
title('SVM classifier result')
figure()
confusionchart(testLabels,treeLabels,'RowSummary','total-normalized');
title('Decision tree classifier result')
%% compare LDA SVM and decision tree
clear all; close all; clc;
num1 = 0;
num2 = 4;

[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
train_images = im2double(train_images);
imgSize = size(train_images);
train_images = reshape(train_images,[imgSize(1) * imgSize(2), imgSize(3)]);

[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImages = im2double(testImages);
testImagesSize = size(testImages);
testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

selectedTrainImages = [train_images(:,train_labels==num1) train_images(:,train_labels==num2)];
selectedTrainLabels = [train_labels(train_labels==num1); train_labels(train_labels==num2)];

selectedTestImages = [testImages(:,testLabels==num1) testImages(:,testLabels==num2)];
selectedTestLabels = [testLabels(testLabels==num1); testLabels(testLabels==num2)];

% LDA already done
% [4,9,0.947262682069312]
% [0,4,0.998470948012233]

% SVM on 0,4
Mdl = fitcecoc(selectedTrainImages',selectedTrainLabels);
test_labels = predict(Mdl,selectedTestImages');
trainlabels = predict(Mdl,selectedTrainImages');
easiestSVMaccuracyTrain = sum(selectedTrainLabels==trainlabels)/length(trainlabels);
easiestSVMaccuracyTest = sum(selectedTestLabels==test_labels)/length(test_labels);
% decision tree on 0,4
tree=fitctree(selectedTrainImages',selectedTrainLabels);
test_labels = predict(tree,selectedTestImages');
trainlabels = predict(tree,selectedTrainImages');
easiestDTaccuracyTrain = sum(selectedTrainLabels==trainlabels)/length(trainlabels);
easiestDTaccuracyTest = sum(selectedTestLabels==test_labels)/length(test_labels);

num1 = 4;
num2 = 9;

[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
train_images = im2double(train_images);
imgSize = size(train_images);
train_images = reshape(train_images,[imgSize(1) * imgSize(2), imgSize(3)]);

[testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
testImages = im2double(testImages);
testImagesSize = size(testImages);
testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

selectedTrainImages = [train_images(:,train_labels==num1) train_images(:,train_labels==num2)];
selectedTrainLabels = [train_labels(train_labels==num1); train_labels(train_labels==num2)];

selectedTestImages = [testImages(:,testLabels==num1) testImages(:,testLabels==num2)];
selectedTestLabels = [testLabels(testLabels==num1); testLabels(testLabels==num2)];

% LDA already done
% [4,9,0.947262682069312]
% [0,4,0.998470948012233]

% SVM on 4,9
Mdl = fitcecoc(selectedTrainImages',selectedTrainLabels);
test_labels = predict(Mdl,selectedTestImages');
trainlabels = predict(Mdl,selectedTrainImages');
hardestSVMaccuracyTrain = sum(selectedTrainLabels==trainlabels)/length(trainlabels);
hardestSVMaccuracyTest = sum(selectedTestLabels==test_labels)/length(test_labels);
% decision tree on 4,9
tree=fitctree(selectedTrainImages',selectedTrainLabels);
test_labels = predict(tree,selectedTestImages');
trainlabels = predict(tree,selectedTrainImages');
hardestDTaccuracyTrain = sum(selectedTrainLabels==trainlabels)/length(trainlabels);
hardestDTaccuracyTest = sum(selectedTestLabels==test_labels)/length(test_labels);
%%
X = categorical({'Easiest train','Hardest train', 'Easiest test','Hardest test'});
X = reordercats(X,{'Easiest train','Hardest train', 'Easiest test','Hardest test'});
Y = [0.994900127496813 easiestSVMaccuracyTrain easiestDTaccuracyTrain;
    0.955304893562887 hardestSVMaccuracyTrain hardestDTaccuracyTrain;
    0.998470948012233 easiestSVMaccuracyTest easiestDTaccuracyTest;
    0.947262682069312 hardestSVMaccuracyTest hardestDTaccuracyTest];
h = bar(X,Y)
ylim([0.9 1.1])
ylabel('Accuracy 0.9-1.1')
set(h, {'DisplayName'}, {'Linear','Decision Tree','SVM'}')
legend()
set(gca,'Fontsize',20)
%%
function [threshold,w,sortOne,sortZero] = digits_trainer_2(num1, num2, feature, Y, labels)
    draw = num1==4 & num2 ==9;
    one = Y(1:feature,labels==num1);
    zero = Y(1:feature,labels==num2);
    mo = mean(one,2);
    mz = mean(zero,2);
    [~, no] = size(one);
    [~, nz] = size(zero);
    Sw = 0; % within class variances
    for k = 1:no
        Sw = Sw + (one(:,k) - mo)*(one(:,k) - mo)';
    end
    for k = 1:nz
       Sw =  Sw + (zero(:,k) - mz)*(zero(:,k) - mz)';
    end

    Sb = (mo-mz)*(mo-mz)'; % between class

    [V2, D] = eig(Sb,Sw); % linear disciminant analysis
    [lambda, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    vOne = w'*one;
    vZero = w'*zero;
    
    if mean(vOne) > mean(vZero)
        w = -w;
        vOne = -vOne;
        vZero = -vZero;
    end

    sortOne = sort(vOne);
    sortZero = sort(vZero);

    t1 = length(sortOne);
    t2 = 1;
    while sortOne(t1) > sortZero(t2)
        t1 = t1 - 1;
        t2 = t2 + 1;
    end
    threshold = (sortOne(t1) + sortZero(t2))/2;
    
    if draw
        figure(4)
        plot(vOne,zeros(no),'ob','Linewidth',2)
        hold on
        plot(vZero,ones(nz),'dr','Linewidth',2)
        ylim([0 1.2])
        xlabel('Pval')
        ylabel('Two digits')

        figure(5)
        subplot(1,2,1)
        histogram(sortOne,30); hold on, plot([threshold threshold], [0 1000], 'r')
        set(gca,'Fontsize',14)
        title(sprintf('NUM: %s',string(num1)))
        xlabel('Pval')
        ylabel('Frequency')
        subplot(1,2,2)
        histogram(sortZero,30); hold on, plot([threshold threshold], [0 1000], 'r')
        set(gca,'Fontsize',14)
        title(sprintf('NUM: %s',string(num2)))
        xlabel('Pval')
        ylabel('Frequency')
    end
end

function accuracy = classify_3(num1, num2, num3)
    [train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    train_images = im2double(train_images);
    imgSize = size(train_images);
    train_images = reshape(train_images,[imgSize(1) * imgSize(2), imgSize(3)]);
    
    [testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    testImages = im2double(testImages);
    testImagesSize = size(testImages);
    testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

    selectedImages1 = [train_images(:,train_labels==num1) train_images(:,train_labels==num2) train_images(:,train_labels==num3)];
    num1OrNotLabels = [train_labels(train_labels==num1); -ones(sum(train_labels==num2),1); -ones(sum(train_labels==num3),1)];
    
    selectedImages2 = [train_images(:,train_labels==num2) train_images(:,train_labels==num3)];
    num2OrNotLabels = [train_labels(train_labels==num2); train_labels(train_labels==num3)];
    
    Mdl1 = fitclinear(selectedImages1',num1OrNotLabels);
    Mdl2 = fitclinear(selectedImages2',num2OrNotLabels);
    
    selectedTestImages = [testImages(:,testLabels==num1) testImages(:,testLabels==num2) testImages(:,testLabels==num3)];
    selectedTestLabels = [testLabels(testLabels==num1); testLabels(testLabels==num2); testLabels(testLabels==num3)];
    test_labels1 = predict(Mdl1,selectedTestImages');
    test_labels2 = predict(Mdl2,selectedTestImages');
    for j = 1:length(test_labels1)
        if test_labels1(j) < 0
            test_labels1(j) = test_labels2(j);
        end
    end

    accuracy = sum(selectedTestLabels==test_labels1)/length(test_labels1);
end

function accuracy = checkAccuracy(num1, num2, feature, U, threshold, w)
    [testImages, testLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
    testImages = im2double(testImages);
    testImagesSize = size(testImages);
    testImages = reshape(testImages,[testImagesSize(1) * testImagesSize(2), testImagesSize(3)]);

    selectedImages = [testImages(:,testLabels==num1) testImages(:,testLabels==num2)];
    selectedLabels = [testLabels(testLabels==num1); testLabels(testLabels==num2)];
    U = U(:,1:feature); % Add this in
    IMat = U'*selectedImages; % PCA projection
    pval = w'*IMat;

    result = [];
    for n = 1 : length(pval)
        if pval(n) > threshold
            result = [result num2];
        else
            result = [result num1];
        end
    end
    accuracy = sum(selectedLabels'==result)/length(result);

end

function accuracy = checkTrainAccuracy(num1, num2, feature, U, threshold, w)
    [train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
    train_images = im2double(train_images);
    imgSize = size(train_images);
    train_images = reshape(train_images,[imgSize(1) * imgSize(2), imgSize(3)]);

    selectedImages = [train_images(:,train_labels==num1) train_images(:,train_labels==num2)];
    selectedLabels = [train_labels(train_labels==num1); train_labels(train_labels==num2)];
    U = U(:,1:feature); % Add this in
    IMat = U'*selectedImages; % PCA projection
    pval = w'*IMat;

    result = [];
    for n = 1 : length(pval)
        if pval(n) > threshold
            result = [result num2];
        else
            result = [result num1];
        end
    end
    accuracy = sum(selectedLabels'==result)/length(result);

end
