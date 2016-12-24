%% This script implement the custom fourier transform, 
function y_padded = cfft(x, dimension, paddings) 
	% Example of calling
	% y = cfft(x, [1, 2, 3], [64, 64, 0, 0, 0, 0]) 


	% Pad the signal with paddings
	x_padded = x;
	
	x_padded = padarray(x_padded, paddings, 'both'); % Padding with zeros, this will increase the frequency resolution
	
	%size(x_padded)
	
	y_padded = x_padded;
	for dim=dimension
		% Take n points Fourier transform along 
		y_padded = fft(y_padded, [], dim);
	end
end