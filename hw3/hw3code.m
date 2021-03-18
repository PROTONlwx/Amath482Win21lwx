%% task1
% Clean workspace
clear all; close all; clc

load('cam1_1.mat')
numFrames1_1 = size(vidFrames1_1,4);

positions1_1 = zeros(2, numFrames1_1);
for j = 1:numFrames1_1
    X = rgb2gray(vidFrames1_1(:,:,:,j)); %y,x
    X((1:480),(1:250)) = zeros(480,250);
    X((1:480),(451:640)) = zeros(480,190);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions1_1(:, j) = [bigx; bigy];
end

load('cam2_1.mat')
numFrames2_1 = size(vidFrames2_1,4);

positions2_1 = zeros(2, numFrames2_1-58);
for j = 20:245
    X = rgb2gray(vidFrames2_1(:,:,:,j)); %y,x
    X((1:480),(1:220)) = zeros(480,220);
    X((1:480),(421:640)) = zeros(480,220);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions2_1(:, j-19) = [bigx; bigy];
end

load('cam3_1.mat')
numFrames3_1 = size(vidFrames3_1,4);

positions3_1 = zeros(2, numFrames3_1-7);
for j = 7:numFrames3_1
    X = rgb2gray(vidFrames3_1(:,:,:,j)); %y,x
    X((1:230),(1:640)) = zeros(230,640);
    X((351:480),(1:640)) = zeros(130,640);
    X((1:480),(1:170)) = zeros(480,170);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions3_1(:, j-6) = [bigx; bigy];
end
% first plot three cameras' x and y motion

position_1 = [positions1_1; positions2_1; positions3_1];
plotEverything(position_1, 226)
%% task2
% Clean workspace
clear all; close all; clc

load('cam1_2.mat')
numFrames1_2 = size(vidFrames1_2,4);

positions1_2 = zeros(2, numFrames1_2-14);
for j = 15:numFrames1_2
    X = rgb2gray(vidFrames1_2(:,:,:,j)); %y,x
    X((1:480),(1:250)) = zeros(480,250);
    X((1:480),(451:640)) = zeros(480,190);
    X((1:200),(1:640)) = zeros(200,640);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions1_2(:, j-14) = [bigx; bigy];
end

load('cam2_2.mat')
numFrames2_2 = size(vidFrames2_2,4);

positions2_2 = zeros(2, numFrames2_2-56);
for j = 1:numFrames2_2-56
    X = rgb2gray(vidFrames2_2(:,:,:,j)); %y,x
    X((1:480),(1:180)) = zeros(480,180);
    X((1:480),(421:640)) = zeros(480,220);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions2_2(:, j) = [bigx; bigy];
end


load('cam3_2.mat')
numFrames3_2 = size(vidFrames3_2,4);

positions3_2 = zeros(2, numFrames3_2-27);
for j = 15:numFrames3_2-13
    X = rgb2gray(vidFrames3_2(:,:,:,j)); %y,x
    X((1:200),(1:640)) = zeros(200,640);
    X((331:480),(1:640)) = zeros(150,640);
    X((1:480),(1:220)) = zeros(480,220);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions3_2(:, j-14) = [bigx; bigy];
end
position_2 = [positions1_2; positions2_2; positions3_2];
plotEverything(position_2, 300)

%% task3
% Clean workspace
clear all; close all; clc

load('cam1_3.mat')
numFrames1_3 = size(vidFrames1_3,4);

positions1_3 = zeros(2, numFrames1_3-9);
for j = 10:numFrames1_3
    X = vidFrames1_3(:,:,:,j); %y,x
    X((1:480),(1:250)) = zeros(480,250);
    X((1:480),(451:640)) = zeros(480,190);
    X((1:200),(1:640)) = zeros(200,640);
    %keep red component
    X(:,:,(2:3)) = zeros(480,640, 2);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions1_3(:, j-9) = [480-bigx; bigy];
end

load('cam2_3.mat')
numFrames2_3 = size(vidFrames2_3,4);

positions2_3 = zeros(2, numFrames2_3-51);
for j = 1:numFrames2_3-51
    X = rgb2gray(vidFrames2_3(:,:,:,j)); %y,x
    X((1:480),(1:180)) = zeros(480,180);
    X((1:480),(421:640)) = zeros(480,220);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions2_3(:, j) = [bigx; bigy];
end

load('cam3_3.mat')
numFrames3_3 = size(vidFrames3_3,4);

positions3_3 = zeros(2, numFrames3_3-7);
for j = 1:numFrames3_3-7
    X = rgb2gray(vidFrames3_3(:,:,:,j)); %y,x
    X((1:200),(1:640)) = zeros(200,640);
    X((331:480),(1:640)) = zeros(150,640);
    X((1:480),(1:220)) = zeros(480,220);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions3_3(:, j) = [bigx; bigy];
end

position_3 = [positions1_3; positions2_3; positions3_3];
plotEverything(position_3, 230)
%% task4
% Clean workspace
clear all; close all; clc

load('cam1_4.mat')
numFrames1_4 = size(vidFrames1_4,4);

positions1_4 = zeros(2, numFrames1_4);
for j = 1:numFrames1_4
    X = vidFrames1_4(:,:,:,j); %y,x
    X((1:480),(1:250),:) = zeros(480,250,3);
    X((1:480),(451:640),:) = zeros(480,190,3);
    X((1:200),(1:640),:) = zeros(200,640,3);
    %keep red component
    X(:,:,(2:3)) = zeros(480,640, 2);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions1_4(:, j) = [bigx; bigy];
end

load('cam2_4.mat')
numFrames2_4 = size(vidFrames2_4,4);

positions2_4 = zeros(2, numFrames2_4-13);
for j = 11:numFrames2_4-3
    [bw,X] = createMask(vidFrames2_4(:,:,:,j));
    X((1:480),(1:180),:) = zeros(480,180,3);
    X((1:480),(401:640),:) = zeros(480,240,3);
    X((401:480),(1:640),:) = zeros(80,640,3);
    X((1:130),(1:640),:) = zeros(130,640,3);
    X = rgb2gray(X);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    X((bigy-10:bigy+9),(bigx-10:bigx+9),:) = ones(20,20)*255;
    %imshow(X)
    
    positions2_4(:, j-10) = [bigx; bigy];
end

load('cam3_4.mat')
numFrames3_4 = size(vidFrames3_4,4);

positions3_4 = zeros(2, numFrames3_4-2);
for j = 3:numFrames3_4
    X = vidFrames3_4(:,:,:,j); %y,x
    X((1:150),(1:640),:) = zeros(150,640,3);
    X((301:480),(1:640),:) = zeros(180,640,3);
    X((1:480),(1:220),:) = zeros(480,220,3);
    %keep red component
    X(:,:,(2:3)) = zeros(480,640, 2);
    M = max(abs(X),[],'all');
    [bigy,bigx] = ind2sub(size(X),find(abs(X) == M));
    %imshow(X)
    bigx = floor(mean(bigx));
    bigy = floor(mean(bigy));
    positions3_4(:, j-2) = [bigx; bigy];
end

position_4 = [positions1_4; positions2_4; positions3_4];
plotEverything(position_4, 392)
%%
function plotEverything(positions, data_length)
    positions = bsxfun(@minus,positions,mean(positions, 2));
    % first plot three cameras' x and y motion
    figure(1)
    tiledlayout(3,1)
    nexttile
    plot([1:data_length], positions(1,:), 'displayname', 'horizontal', 'linewidth', 3), hold on
    plot([1:data_length], positions(2,:), 'displayname', 'vertical', 'linewidth', 3)
    title('Camera_1')
    set(gca,'Fontsize',24)
    legend();
    nexttile
    plot([1:data_length], positions(3,:), 'displayname', 'horizontal', 'linewidth', 3), hold on
    plot([1:data_length], positions(4,:), 'displayname', 'vertical', 'linewidth', 3)
    title('Camera_2')
    set(gca,'Fontsize',24)
    legend();
    nexttile
    plot([1:data_length], positions(5,:), 'displayname', 'horizontal', 'linewidth', 3), hold on
    plot([1:data_length], positions(6,:), 'displayname', 'vertical', 'linewidth', 3)
    title('Camera_3')
    set(gca,'Fontsize',24)
    legend();

    % plot the singular values
    [~,n]=size(positions);
    [U,S,V] = svd(positions'/sqrt(n-1), 'econ');
    lambda =diag(S).^2;
    figure(2)
    plot(lambda / sum(lambda),'ko','Linewidth',2)
    title('Energy of diagonal variances')
    set(gca,'Fontsize',16)
    
    figure(3)
    Y = V' * positions;
    plot([1:data_length], Y(1, :), [1:data_length], Y(2, :), [1:data_length], Y(3, :), 'linewidth',2)
    title('Projection of displacement data on principal components')
    set(gca,'Fontsize',16)
    legend('PC1', 'PC2','PC3', 'PC4')
end

function [BW,maskedRGBImage] = createMask(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 18-Feb-2021
%------------------------------------------------------


% Convert RGB image to chosen color space
I = rgb2hsv(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.935;
channel1Max = 0.900;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.000;
channel2Max = 1.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.000;
channel3Max = 1.000;

% Create mask based on chosen histogram thresholds
sliderBW = ( (I(:,:,1) >= channel1Min) | (I(:,:,1) <= channel1Max) ) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Invert mask
BW = ~BW;

% Initialize output masked image based on input image.
maskedRGBImage = RGB;

% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

end
