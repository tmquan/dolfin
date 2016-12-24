%% This script implement the custom fourier transform, 
function y_cropped = cfft(x, dimension, paddings) 
	% Example of calling
	% y = cfft(x, [1, 2, 3], [64, 64, 0, 0, 0, 0]) 

	shape = size(x);
	y = x;
	for dim=dimension
		% Take n points Fourier transform along 
		y = ifft(y, [], dim);
	end

	% Calculate the new shape after cropping
	oldshape = size(y);

	newshape = (1+paddings[1] : end - paddings[1],
		  		1+paddings[2] : end - paddings[2],
		  		1+paddings[3] : end - paddings[3],
		  		1+paddings[4] : end - paddings[4],
		  		1+paddings[5] : end - paddings[5],
		  		1+paddings[6] : end - paddings[6]
		  		);
	% Crop the data
	y_cropped = y(newshape);
end