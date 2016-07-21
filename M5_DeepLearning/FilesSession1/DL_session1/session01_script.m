%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---[ I. Basic MatConvNet skills ]--- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Go to the matlab directory: cd (YOUR_DIRECTORY)\DL_session1
% First we run vl_setupnn (change path if necessary)
run ./../../MatConvNet/matlab/vl_setupnn

% Have a closer look at a network
% load AlexNet
load('./../data/nets/imagenet-alex_dagnn.mat');

% Perform inference with alexNet on the MIT data set
% you will see the scores of the most probably class for some of the images
% in the MIT set. Try for some other classes by changing folder in check_alexnet_results
% (Press ^C to stop)
% check_alexnet_results

% check net members of the net. Get familiar with these fields
alexNet

% information about the layers
alexNet.layers(1)   
alexNet.layers(1).block
    
% vars stores all partial results
alexNet.vars     
alexNet.vars(1)
alexNet.vars(2)
alexNet.vars(3)
        
% parames contains the the parameters of the network
alexNet.params(1)
alexNet.params(2)
size(alexNet.params(1).value)
        
% Visualizing net layers and properties: we provide the print function with 
% the information that we input an image of [227,227,3]
alexNet.print({'input',[227,227,3,1]},'MaxNumColumns', 5);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ---[ II. Finetuning of AlexNet on the MIT dataset ]--- %%           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the dataset MIT data set as an IMDB structure which will be used
% as an input to the CNN. To make the testing faster we rescale the images
% to 37x37x3
imdb = createIMDB_RAM('./../data/MIT_split');

% To normalize the images often we subtract the average image of the dataset.
% Because we want to use a pre-trained network, we need to use the same average 
% image to normalize the IMDB
imdb.images.data_mean=imresize(alexNet.meta.normalization.averageImage,[37,37]);

% Have a look at the created MATLAB structure
imdb.images
imdb.meta

% Prepare the data similarly as the data which was used to train AlexNet
% Part of the preparation is to subtract the dataset average image from all
% images. This was found to lead to faster convergence.
imdb = normalizeIMDB(imdb);

% Before we finetune alexNet for this dataset let us have a look the file
% alexnet_train_small (see document II.2)

% Finetune the AlexNet on the dataset. We provide the function with 
%       imdbNorm                    : the prepared dataset
%       './../results/AlexSmall'    : a location to save results for each epoch 
%       alexNet                     : the pretrained network
% We will only run two epochs, later you can try running more epochs.
[alexNetSmall, info] = alexnet_train_small(imdb, './../results/AlexSmall',alexNet)

% to evalute the network on the test set
[acc]=eval_acc(imdb,alexNetSmall)

% Have a look at the alexNetSmall. 
alexNetSmall.print({'input',[37,37,3,1]},'MaxNumColumns', 5);
% Have a look at the dimensionality of the vars (which start with (37x37x3x1))
% and see how the dimensionality changes from one layer to the other. Looking
% at the layer definitions in alexnet_train_small.m  you should be able to 
% exactly follow the changes of the dimensionality. This is very important 
% if you later want to design your own network.

% we will now compute the imdb for images of size 227x227x3
imdb = createIMDB_DISK('./../data/MIT_split');
% again we copy the mean from alexNet
imdb.images.data_mean = alexNet.meta.normalization.averageImage;

% ---[ Exercise 1 ]--- %%
% After finishing exercise 1 (applying the changes to alexnet_train.m), 
% run the function:
[alexNet, info] = alexnet_train(imdb, './../results/Alex',alexNet)

% See if you obtain better results than before
[acc]=eval_acc_DISK(imdb,alexNet)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---[ III. Extracting Features From AlexNet ]--- %%      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Here we will extract features from AlexNet which can for example be used
% in a BOW pipeline or for Image Retrieval

% ---[ Exercise 2 ]--- %%      
% Follow the comments in the document.