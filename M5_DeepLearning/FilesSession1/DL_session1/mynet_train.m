function [net, info] = mynet_train(imdb, expDir,varargin)

% some common options
opts.train.batchSize = 80;
opts.train.numEpochs = 80;
opts.train.continue = false ;   % If set to true training will start where it stopt before (to start from zero use false)
opts.train.gpus = [] ;
opts.train.learningRate = [1e-2*ones(1,25),1e-2*ones(1,25)]; % here we say that after 25 steps the learning rate should go to 1e-3
opts.train.weightDecay = 0.001; %0.001
opts.train.momentum = 0.;   
opts.train.expDir = expDir;
opts.train.numSubBatches = 1;

% getBatch options
bopts.useGpu = numel(opts.train.gpus) >  0 ;

if(numel(varargin) > 0)
    netPre = varargin{1};
end

% network definition!
% MATLAB handle, passed by reference
net = dagnn.DagNN() ;

%% EXERCISE 3: Make your own network
% change the network definition below for a new one and evaluate its performance

net.addLayer('conv1', dagnn.Conv('size', [3 3 3 16], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});
net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [4, 4], 'stride', [4 4], 'pad', [0 0 0 0]), {'relu1'}, {'pool1'}, {});
%Meaning of stride in pooling layer?
net.addLayer('conv2', dagnn.Conv('size', [3 3 16 16], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'relu2'}, {});
net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [4, 4], 'stride', [4 4], 'pad', [0 0 0 0]), {'relu2'}, {'pool2'}, {});

net.addLayer('conv3', dagnn.Conv('size', [3 3 16 16], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});
net.addLayer('pool3', dagnn.Pooling('method', 'max', 'poolSize', [11, 11], 'stride', [1 1], 'pad', [0 0 0 0]), {'relu3'}, {'pool3'}, {});

net.addLayer('classifier', dagnn.Conv('size', [1 1 16 8], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool3'}, {'classifier'},  {'conv4f'  'conv4b'});
%Why 8 not 16?
net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
% -- end of the network

%% EXERCISE 4: Use Drop-out and Batchnormalization in your network
% In troduce dropout layers batchnormalization layers in your network and evaluate performance
% normally they are not used together
% also try different initializations of the network
%
% net.addLayer('drop1', dagnn.DropOut('rate', 0.3), {'pool1'}, {'drop1'}, {});
% net.addLayer('bn1', dagnn.BatchNorm('numChannels', 24), {'pool1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});

% initialization of the weights (CRITICAL!!!!)
if(numel(varargin) > 0)
    initNet_FineTuning(net, netPre);
else
    %% change the initialization 
    %initNet_He(net);
    %initNet_xavier(net);
    initNet(net, 1/100);
end

% do the training!
[net,info] = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
end

% getBatch for IMDBs fully in RAM
function inputs = getBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch);
labels = imdb.images.labels(1,batch) ;
if opts.useGpu > 0
    images = gpuArray(images) ;
end

inputs = {'input', images, 'label', labels} ;
end

% getBatch for IMDBs that are too big to be in RAM
function inputs = getBatchDisk(opts, imdb, batch)
cell_images = vl_imreadjpeg(imdb.images.filenames(batch),'numThreads', 3) ;
[H, W, CH] = size(imdb.images.data_mean);

images = zeros(H, W, CH, 'single');
for i=1:numel(cell_images)
    im = single(imresize(cell_images{i},[H,W]));
    images(:,:,:, i) = im - imdb.images.data_mean;
end

labels = imdb.images.labels(1,batch) ;
if opts.useGpu > 0
    images = gpuArray(images) ;
end

inputs = {'input', images, 'label', labels} ;
end


function initNet_He(net, f)
net.initParams();
for l=1:length(net.layers)
    % is a convolution layer?
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        
        [h,w,in,out] = size(net.params(f_ind).value);
        he_gain = 0.7*sqrt(2/(h*w*in)); % sqrt(1/fan_in)
        net.params(f_ind).value = he_gain*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate = 1;
        net.params(f_ind).weightDecay = 1;
        
        net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate = 0.5;
        net.params(b_ind).weightDecay = 1;
    end
end
end

function initNet_xavier(net)
net.initParams();
for l=1:length(net.layers)
    % is a convolution layer?
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
        f_ind = net.layers(l).paramIndexes(1);
        b_ind = net.layers(l).paramIndexes(2);
        
        [h,w,in,out] = size(net.params(f_ind).value);
        xavier_gain = 0.7*sqrt(1/(h*w*in)) % sqrt(1/fan_in)
        net.params(f_ind).value = xavier_gain*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate = 1;
        net.params(f_ind).weightDecay = 1;
        
        net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate = 0.5;
        net.params(b_ind).weightDecay = 1;
    end
end
end

function initNet(net, f)
	net.initParams();
	%

	f_ind = net.layers(1).paramIndexes(1);
	b_ind = net.layers(1).paramIndexes(2);
	net.params(f_ind).value = 10*f*randn(size(net.params(f_ind).value), 'single');
	net.params(f_ind).learningRate = 1;
	net.params(f_ind).weightDecay = 1;

	for l=2:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			net.params(f_ind).value = f*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = f*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
		end
	end
end

