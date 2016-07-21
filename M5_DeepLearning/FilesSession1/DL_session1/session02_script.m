% Copy these files to the same directory as the matlab files from last week
% Copy the file BatchNorm.m to the directory:
% (YOUR_DIRECTORY)/matconvnet-1.0-beta17/matlab/+dagnn/
%
% Go to the matlab directory: cd (YOUR_DIRECTORY)\DL_session1
% overwrite the existing file BatchNorm.m

% Run vl_setupnn (change path if necessary)
run ./../MatConvNet/matlab/vl_setupnn

% Load the dataset MIT data set as an IMDB structure which will be used
% as an input to the CNN. To make the testing faster we rescale the images
% to 37x37x3
imdb = createIMDB_RAM('./../data/MIT_split');
imdb = normalizeIMDB(imdb);

% Run the code
[mynet, info] = mynet_train(imdb, './../results/mynet5')

% Start with Exercise 3 and when done Exercise 4

