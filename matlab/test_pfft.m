% Test custom cfft;
clc; clear all; close all;

% Load the images
oldshape = [512, 512, 1, 1, 1, 6];
paddings = [64, 64, 0, 0, 0, 0];

imgs = imreadtif('std.tif');
imgs = single(imgs);

imgs = reshape(imgs, oldshape);
freq = pfft(imgs, [1, 2], paddings);
imgs = pifft(freq, [1, 2], paddings);