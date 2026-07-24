function [mtf2D] = mtf_2D(mtfAxis, mtfVal, N, uSize, showFlag)
% mtfAxis in cycles/mm, mtfVal corresponding values, uSize in mm/pixel

pixSize = uSize(1);                      % mm per pixel
% frequency vector in cycles/mm, symmetric and centered so DC at center
fx_mm = ((-floor(N/2):ceil(N/2)-1) / N) / pixSize;  
[FX, FY] = meshgrid(fx_mm, fx_mm);

% radial frequency in cycles/mm (DC at FX=FY=0 -> center)
R = sqrt(FX.^2 + FY.^2);

% interpolate 1D MTF (extrapolate beyond input axis -> 0)
mtf2D = interp1(mtfAxis(:), mtfVal(:), R, 'linear', 0);

if showFlag
    figure;
    imagesc(fx_mm*pixSize, fx_mm*pixSize, mtf2D);
    axis image xy;
    xlabel('f_x (cycles/pixel)'); ylabel('f_y (cycles/pixel)');
    title('2D Radial MTF (cycles/pixel) — DC at center');
    colorbar;
end
end