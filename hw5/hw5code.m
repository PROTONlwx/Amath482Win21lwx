%% task1
% Clean workspace
clear all; close all; clc
v = VideoReader('monte_carlo_low.mp4'); %ski_drop_low monte_carlo_low

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

[U, Sigma, V] = svd(X1,'econ');

rank = 20; %20 for skier video, 75 for car video.
U_R = U(:, 1:rank);
Sigma_R = Sigma(1:rank, 1:rank);
V_R = V(:, 1:rank);

S = U_R'*X2*V_R*diag(1./diag(Sigma_R));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U_R*eV;

zeroOmegas = find(abs(omega) < 0.01);
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions

u_modes = zeros(length(zeroOmegas),length(t));
for iter = 1:length(t)
    u_modes(:,iter) = y0(zeroOmegas, 1).*exp(omega(zeroOmegas, 1)*t(iter));
end
X_low_rank_dmd = Phi(:, zeroOmegas)*u_modes;

X_sparse = X - abs(X_low_rank_dmd);
%R = X_sparse .* (X_sparse < 0);
%X_low_rank_dmd = R + abs(X_low_rank_dmd);
%X_sparse = X_sparse - R;

%%
uToVideo1 = uint8(reshape(X_low_rank_dmd, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]));
vw = VideoWriter('carRace_back');
open(vw)
for j = 1:v.NumFrames
    writeVideo(vw,mat2gray(uToVideo1(:,:,j)));
end
close(vw)

%X_sparse = filter2(fspecial('sobel'), X_sparse .* -1);
uToVideo2 = reshape(X_sparse, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]);
vw = VideoWriter('carRace_obj');
open(vw)
for j = 1:v.NumFrames
    writeVideo(vw,mat2gray(uToVideo2(:,:,j)));
end
close(vw)

%%
slices = [1 floor(v.NumFrames/2) v.NumFrames];
uToVideo1 = uint8(reshape(X_low_rank_dmd, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]));
uToVideo2 = reshape(X_sparse, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]);
uToVideo3 = reshape(X, [sizeFrames(1), sizeFrames(2), sizeFrames(4)]);

figure()

for j = 1:3
    subplot(3 ,3 , 3*(j-1)+1)
    imshow(mat2gray(uToVideo1(:,:,slices(j))));
    title(sprintf('Frame%s: background',string(slices(j))))
    subplot(3 ,3 , 3*(j-1)+2)
    imshow(mat2gray(uToVideo2(:,:,slices(j))));
    title(sprintf('Frame%s: foreground',string(slices(j))))
    subplot(3 ,3 , 3*(j-1)+3)
    imshow(mat2gray(uToVideo3(:,:,slices(j))));
    title(sprintf('Frame%s: Original',string(slices(j))))
end
 ha=get(gcf,'children');
 set(ha(1),'position',[0 .66 .33 .33])%left bottom width height
 set(ha(2),'position',[.33 .66 .33 .33])
 set(ha(3),'position',[.66 .66 .33 .33])
 set(ha(4),'position',[0 .33 .33 .33])
 set(ha(5),'position',[.33 .33 .33 .33])
 set(ha(6),'position',[.66 .33 .33 .33])
 set(ha(7),'position',[0 0 .33 .33])
 set(ha(8),'position',[.33 0 .33 .33])
 set(ha(9),'position',[.66 0 .33 .33])
%% Plotting Eigenvalues (omega)

% make axis lines
line = -15:15;

plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega)*dt,imag(omega)*dt,'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
set(gca,'FontSize',16,'Xlim',[-1.5 0.5],'Ylim',[-3 3])

%%
plot(diag(Sigma),'ko','Linewidth',1)
set(gca,'Fontsize',16)
title('Singular Values')
xlabel('singular values')
ylabel('magnitude')

%%
plot(cumsum(diag(Sigma)./sum(diag(Sigma))),'ko','Linewidth',1), hold on
plot([0 454],[0.85, 0.85], '-r', 'linewidth', 2)
set(gca,'Fontsize',16)
xlabel('singular values')
ylabel('cumulative energy')
%%
tmp = Phi';
waterfall([1:sizeFrames(1) * sizeFrames(2)],1:10,abs(tmp(1:10, :))), colormap([0 0 0])
xlabel('x')
ylabel('modes')
zlabel('|u|')
title('DMD Modes')
set(gca,'FontSize',16)















