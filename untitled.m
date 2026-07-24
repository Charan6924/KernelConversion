clear all

data_C = dicomread("PhantomC\I20");
data_D = dicomread("PhantomD\I20");
info = dicominfo("PhantomC\I20");
mtf_c = load("I20_Kernel_C_MTF_Results_mat.mat");
mtf_c = mtf_c.results;
mtf_d = load("I20_Kernel_D_MTF_Results_mat.mat");
mtf_d = mtf_d.results;
%%
uSize = mtf_d.pixelSize;
figure;plot(mtf_c.mtfAxis, mtf_c.mtfVal);
hold on;
plot(mtf_d.mtfAxis, mtf_d.mtfVal);
xlim([0 2*1/(2*uSize(1))]);
%%
% nyquist frequency
nf = mtf_c.mtfAxis;
nyq_cpm = 0.5 / uSize;           % Nyquist in cycles/mm
% Trim mtf_c to <= Nyquist
idx_c = mtf_c.mtfAxis <= nyq_cpm;
mtf_c_trimAxis = mtf_c.mtfAxis(idx_c);
mtf_c_trimVal  = mtf_c.mtfVal(idx_c);

% Trim mtf_d to <= Nyquist
% So here we truncate the MTF
% and these are the data points we should compare to our model output
idx_d = mtf_d.mtfAxis <= nyq_cpm;
mtf_d_trimAxis = mtf_d.mtfAxis(idx_d);
mtf_d_trimVal  = mtf_d.mtfVal(idx_d);

figure;
subplot(2,1,1)
plot(mtf_c.mtfAxis, mtf_c.mtfVal);
title('Before trunation, full range')
xlabel("f_x (cycles/mm)");
subplot(2,1,2)
plot(mtf_d_trimAxis, mtf_d_trimVal);
title('after trunation, 1xNyquist frequency')
xlabel("f_x (cycles/mm)");



[mtf_c_2D] = mtf_2D(mtf_c_trimAxis, mtf_c_trimVal, 512, uSize, true);
[mtf_d_2D] = mtf_2D(mtf_d_trimAxis, mtf_d_trimVal, 512, uSize, true);
%
%%
% get the ratio mtf
% caveat: dont use this directly, this is very unstable
% I add some regularization to the ratio filter as you can see
% probably do the same thing for our model output
mtfratio_c_d = mtf_c_2D./mtf_d_2D;
mtfratio_d_c = mtf_d_2D./mtf_c_2D;

% plot ratio
fx_pixel = ((-floor(512/2):ceil(512/2)-1) / 512);  
figure;imagesc(fx_pixel,fx_pixel,mtfratio_c_d);
figure;imagesc(fx_pixel,fx_pixel,mtfratio_d_c);


% get the frequency domain of images
% Compute 2-D FFT and center DC
F_c = fftshift(fft2(double(data_C)));
F_d = fftshift(fft2(double(data_D)));

% Plot magnitude with DC at center (cycles/pixel)
figure;
imagesc(fx_pixel, fx_pixel, abs(F_c));
figure;
imagesc(fx_pixel, fx_pixel, abs(F_d));
%% Build regularized MTF conversion filters
% C from D:
%     F_C approximately equals F_D * MTF_C / MTF_D
%
% Regularized inverse:
%     H_D_to_C = MTF_C * MTF_D /
%                (MTF_D^2 + lambda)

lambda = 1e-3 * max(mtf_d_2D(:).^2);

H_D_to_C = (mtf_c_2D .* mtf_d_2D) ./ ...
           (mtf_d_2D.^2 + lambda);

% D from C
lambdaReverse = 1e-3 * max(mtf_c_2D(:).^2);

H_C_to_D = (mtf_d_2D .* mtf_c_2D) ./ ...
           (mtf_c_2D.^2 + lambdaReverse);

% Optional gain limit to suppress extreme high-frequency amplification
maxGain = 5; % the max our data is around 3.6 i think.
H_D_to_C = min(H_D_to_C, maxGain);
H_C_to_D = min(H_C_to_D, maxGain);

figure;
imagesc(fx_pixel, fx_pixel, H_D_to_C);
axis image;
colorbar;
xlabel("f_x (cycles/pixel)");
ylabel("f_y (cycles/pixel)");
title("Regularized MTF filter: D to C");
%%
% before we do the reconstruction let us compute the ratio directly from
% the F_c and F_d and compare with mtfratio_c_d and mtfratio_d_c
%% Empirical Fourier-domain ratio
% This is meaningful only when the images represent the same object and are
% spatially registered. Use complex regularized division. 
% it is a trick to do for the phantom 

fftLambda = 1e-6 * max(abs(F_d(:)).^2);

H_empirical_D_to_C = ...
    F_c .* conj(F_d) ./ (abs(F_d).^2 + fftLambda);

figure;
imagesc(fx_pixel, fx_pixel, log1p(abs(H_empirical_D_to_C)));
axis image;
colorbar;
xlabel("f_x (cycles/pxel)");
ylabel("f_y (cycles/pxel)");
title("Empirical |F_C/F_D|, regularized");
%%
% now we can try to use the mtfratio_c_d and mtfratio_d_c to reconstruct
% image from one to the other
F_reconstructed_C = F_d .* H_D_to_C;
reconstructed_C = real(ifft2(ifftshift(F_reconstructed_C)));

F_reconstructed_D = F_c .* H_C_to_D;
reconstructed_D = real(ifft2(ifftshift(F_reconstructed_D)));

%% Display results
displayLimits = prctile([data_C(:); data_D(:)], [1, 99]);

figure;
imagesc(data_C, displayLimits);
axis image off;
colormap gray;
colorbar;
title("Original image C");

figure;
imagesc(data_D, displayLimits);
axis image off;
colormap gray;
colorbar;
title("Original image D");

figure;
imagesc(reconstructed_C, displayLimits);
axis image off;
colormap gray;
colorbar;
title("Reconstructed C from D");

figure;
imagesc(reconstructed_D, displayLimits);
axis image off;
colormap gray;
colorbar;
title("Reconstructed D from C");