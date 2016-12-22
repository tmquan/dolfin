%% This file is the demo script of the CSC with arranging the tensor shape
clc; clear all; close all;

%% Reproducible seeding
rng(2017);

%% Define the layer parameters
dictSize  = 64; % Number of atoms in the dictionary
dataSize  = 6;  % Number of images in the entire dataset

%% Define the inner tensor shape
atomInner = [15,   15,  1]; % For grayscale atom, it is 15x15x1
dataInner = [256, 256,  1]; % For grayscale image, it is 256x256x1

%% Define the outer tensor shape
atomOuter = [atomInner, dictSize,        1]; 	% original dictionary size, before padding
dictOuter = [dataInner, dictSize,        1]; 	% dimy * dimx * dimz * dimk * 1, zeropadding
dataOuter = [dataInner,        1, dataSize]; 	% dimy * dimx * dimz *    1 * dimn
blobOuter = [dataInner, dictSize, dataSize]; 	% dimy * dimx * dimz * dimk * 1, zeropadding

%% Define a bunch of parameters, if we run on the graph base, needs to wrap by a class of parameters
Lambda   = 0.1;
Rho      = 0.001;
Sigma    = 0.001;

Jxp      = 10;
mu_x     = 0.01;
tau_x    = 2;

Jdp      = 10;
mu_d     = 0.01;
tau_d    = 2;

alpha_x  = 1.8; % Relaxation parameter of splitting sparse maps
alpha_d  = 1.8; % Relaxation parameter of splitting dictionary

Jmax     = 100;  % Maximum number of iteration
eps_abs  = 1e-6; % Absolute stopping tolerances
eps_rel  = 1e-6; % Relative stopping tolerancces

fixed_d  = false;
fixed_x  = false; % Control parameters of whether we want to optimize the dictionary of the sparse code

%% Initialization, Preallocate the layer placeholder, 5D tensorshape
Si   = complex(zeros(dataOuter)); % dimy * dimx * dimz * 1 * dimn; data placeholder
Sf   = complex(zeros(dataOuter)); % dimy * dimx * dimz * 1 * dimn; freq placeholder 

%% PrecomputeFetch the image data to S and perform custom fourier transform
% images should be normalized within the range [0, 1]
S0   = images;

%% Initialize other related splitting variables for sparse codes
Yi   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; current sparse code @ iteration i
Yf   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; current sparse code in Fourier domain
Yp   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; previous sparse code 
Ui   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; dual variable U @ iteration i

Xi   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; current sparse code @ iteration i
Xf   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; current sparse code in Fourier domain
Xr   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; relaxed sparse code
Uz   = complex(zeros(blobOuter)); % dimy  * dimx * dimz * dimk * dimn; residual of X and Y, in Fourier domain

%% Initialize other related splitting vvariables for dictionary learning
Gi   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; current dictionary @ iteration i
Gf   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; current dictionary in Fourier domain
Gp   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; previous dictionary 
Hi   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; dual variable H @ iteration i

Di   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; current dictionary @ iteration i
Df   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; current dictionary in Fourier domain
Dr   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; relaxed dictionary 
Hz   = complex(zeros(dictOuter)); % dimy  * dimx * dimz * dimk * 1; residual of D and G, in Fourier domain

% That's all for constructing the graph
%%








