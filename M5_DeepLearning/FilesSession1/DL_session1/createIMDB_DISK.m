function imdb = createIMDB_DISK(folder)
% imdb is a matlab struct with several fields, such as:
%	- images: contains data, labels, ids dataset mean, etc.
%	- meta: contains meta info useful for statistics and visualization
%	- any other you want to add

% labels we use train=1, val=2, test=3

imdb = struct();

classes = {'Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding'};
N=2688; % number of images
ValFrac = 0.2;    % fraction of images from train used for validation set.

H=227;W=227;CH=3;   % the common size to which images are resized

% we can initialize part of the structures already
meta.sets = {'train', 'val', 'test'};
meta.classes = classes;

% we do not save the images, they will be read during training
images.data = [];
% this will contain the mean of the training set
images.data_mean = zeros(H, W, CH, 'single');
% a label per image
images.labels = single(zeros(1, N));
% vector indicating to which set an image belong, i.e.,
% training, validation, etc.
images.set = uint8(zeros(1, N));

% loading images, set labels and divide in train, val and test
counter=1;
for ii=1: size(classes,2)
    image_list=dir(sprintf('%s/train/%s/*.jpg',folder,classes{ii}));
    Ntrain=size(image_list,1);
    valtrainsplit=ones(Ntrain,1);
    valtrainsplit(1:round(Ntrain*ValFrac))=2;
    valtrainsplit=valtrainsplit(randperm(length(valtrainsplit)));    % mix values
    for jj=1:Ntrain
        im=single(imresize(imread(sprintf('%s/train/%s/%s',folder,classes{ii},image_list(jj).name)),[H,W]));
        images.filenames{counter} = sprintf('%s/train/%s/%s',folder,classes{ii},image_list(jj).name);
        images.labels(counter) = ii;
        images.set(counter) = valtrainsplit(jj);
        if(valtrainsplit(jj)==1)   % use train image to compute data set mean
            images.data_mean = images.data_mean + im;
        end
        counter=counter+1;
    end
    image_list=dir(sprintf('%s/test/%s/*.jpg',folder,classes{ii}));
    Ntest=size(image_list,1);
    
    for jj=1:Ntest
        images.filenames{counter} = sprintf('%s/test/%s/%s',folder,classes{ii},image_list(jj).name);        
        images.labels(counter) = ii;
        images.set(counter) = 3;    % set test images to 3
        counter=counter+1;
    end
end

meta.rangemultiplier=1;

% let's finish to compute the mean
images.data_mean = images.data_mean ./ sum(images.set(:)==1);

% now let's add some randomization
indices = randperm(N);
images.labels(:) = single(images.labels(indices));
images.set(:) = uint8(images.set(indices));
[images.filenames{:}] = deal(images.filenames{indices});

imdb.meta = meta;
imdb.images = images;

end

