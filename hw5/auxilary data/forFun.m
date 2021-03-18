% Clean workspace
clear all; close all; clc
v = VideoReader('monte_carlo_low.mp4');

dt = 1 / v.FrameRate;
t = 0:dt:v.Duration;
frames = read(v);
sizeFrames = size(frames);
twoDframes = zeros(sizeFrames(4), sizeFrames(1) * sizeFrames(2));
for j = 1:v.NumFrames
    image = rgb2gray(frames(:, :, :, j));
    image = reshape(image, [1, sizeFrames(1) * sizeFrames(2)]);
    twoDframes(j, :) = image;
end
X = twoDframes';
X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X,'econ');

rank = 3;
U_R = U(:, 1:rank);
Sigma_R = Sigma(1:rank, 1:rank);
V_R = V(:, 1:rank);

X_low_rank = U_R * Sigma_R * V_R';
X_sparse = X - abs(X_low_rank);

uToVideo2 = reshape(X_low_rank, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]);
vw = VideoWriter('forfun_back');
open(vw)
for j = 1:v.NumFrames
    writeVideo(vw,mat2gray(uToVideo2(:,:,j)));
end
close(vw)

uToVideo2 = reshape(X_sparse, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]);
vw = VideoWriter('forfun_obj');
open(vw)
for j = 1:v.NumFrames
    writeVideo(vw,mat2gray(uToVideo2(:,:,j)));
end
close(vw)