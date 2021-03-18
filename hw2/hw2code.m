% Clean workspace
clear all; close all; clc
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);

a = 1000;
tau = 0:0.1:tr_gnr;

filter = zeros(size(y));
filter([1:13000+1], 1) = ones(13001, 1);
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   yg = g.*transpose(y);
   ygt = fft(yg) .* transpose(filter);
   ygt_spec(:,j) = fftshift(abs(ygt));
end

figure(1)
pcolor(tau,ks,ygt_spec)
shading interp
set(gca,'ylim',[0, 1000],'xlim',[0, 2.5],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Guitar in GNR');
%%
clear all; close all; clc

[y, Fs] = audioread('Floyd.m4a');
y = y(1:length(y)-1);
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);

a = 2000;
tau = 0:1:tr_gnr;
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   yg = g.*transpose(y);
   ygt = fftshift(abs(fft(yg)));
   ygt_spec(:,j) = ygt;
end

figure(1)
pcolor(tau,ks,log(ygt_spec+1))
shading interp
set(gca,'ylim',[0, 1200],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Floyd Spectrogram');
%%
clear all; close all; clc

[y, Fs] = audioread('Floyd.m4a');
y = y(1:length(y)-1);
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);

a = 2000;
tau = 0:1:tr_gnr;

yt = fft(y);
filter = zeros(size(y));
filter([1:175*60+1], 1) = ones(175*60+1, 1);
ytf = fftshift(yt) .* fftshift(filter);
y2 = fftshift(ifft(ifftshift(ytf)));
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   yg = g.*transpose(y2);
   ygt = fftshift(abs(fft(yg)));
   [M, I] = max(ygt);
   g2 = exp(-0.1*(k - k(I)).^2);
   ygt_spec(:,j) = ygt .* g2;
end

figure(1)
pcolor(tau,ks,log(ygt_spec+1))
shading interp
set(gca,'ylim',[0, 400],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Floyd Bass with Gaussian and Shannon filters');
%%

clear all; close all; clc

[y, Fs] = audioread('0-10.m4a');
%y = y(1:length(y)-1);
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);

a = 800;
tau = 0:0.1:tr_gnr;

yt = fft(y);
filter = zeros(size(y));
filter([300*10: 1200*10], 1) = ones((1200-300)*10+1, 1);
ytf = fftshift(yt) .* fftshift(filter);
y2 = fftshift(ifft(ifftshift(ytf)));
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   yg = g.*transpose(y2);
   ygt = fftshift(abs(fft(yg)));
   %[M, I] = max(ygt);
   %g2 = exp(-0.001*(k - k(I)).^2);
   %ygt_spec(:,j) = ygt .* g2;
   ygt_spec(:,j) = ygt;
end

figure(1)
pcolor(tau,ks,log(ygt_spec+1))
shading interp
set(gca,'ylim',[350, 1200],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title('Floyd Guitar with Gaussian and Shannon filters');






