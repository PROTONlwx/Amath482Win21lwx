% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); y = x; 
z = x;

k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% Average over all 49 data
Untave = zeros(1, n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Untave = Untave + fftn(Un);
end
% Calculate the index of the max frequency in Untave
Untave = abs(fftshift(Untave)/49);
[max_value,idx] = max(Untave(:));
[k0y,k0x,k0z] = ind2sub(size(Untave),idx);

% Use the index of max to find corresponding frequency(k) in x, y, z.
k0x = ks(k0x); k0y = ks(k0y); k0z = ks(k0z);
tau = 0.5;
% Creating Gaussian filters for the three frequencies.
filter_x = exp(-tau*(Kx - k0x).^2); % Define the filter for x
filter_y = exp(-tau*(Ky - k0y).^2); % Define the filter for y
filter_z = exp(-tau*(Kz - k0z).^2); % Define the filter for z
filter_xyz = fftshift(filter_x.*filter_y.*filter_z);
% Create containers of bubble centers.
centers = zeros(49, 3);
% Loop through all realizations to compute the exact submarine positions.
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    % Filter using Gaussion filters.
    Utn = fftn(Un);
    Unft = filter_xyz.*Utn;
    Unf = ifftn(Unft);
    M = max(abs(Unf),[],'all');
    % Get positions with high amplitude.
    [bigy,bigx,bigz] = ind2sub(size(Unf),find(abs(Unf) > M*0.7));
    % Get average and call it the position of submarine.
    centers(j, 1) = mean(bigx)/64*20-10;
    centers(j, 2) = mean(bigy)/64*20-10;
    centers(j, 3) = mean(bigz)/64*20-10;
end
% Plot 3-D path
plot3(centers(:, 1), centers(:, 2), centers(:, 3), 'LineWidth', 3);
xlabel('X');ylabel('Y');zlabel('Z')
set(gca, 'FontSize', 16)
title('3D path')
figure();
% Plot just x-y plane path
plot(centers(:, 1), centers(:, 2), 'LineWidth', 3);
xlabel('X');ylabel('Y')
set(gca, 'FontSize', 16)
title('x-y path')
csvwrite('result.csv',centers);