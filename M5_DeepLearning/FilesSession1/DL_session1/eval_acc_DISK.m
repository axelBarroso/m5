function [acc]=eval_acc_DISK(imdb,net)

% run the CNN
setL=3;  % 3 is the test set

indx=find(imdb.images.set==3);
nIm=length(indx);
scores=zeros(8,nIm);
step=40;

counter=1;
while(counter~=nIm)
    counter2=min(counter+step,nIm);
    
    cell_images = vl_imreadjpeg(imdb.images.filenames(indx(counter:counter2)),'numThreads', 3) ;
    [H, W, CH] = size(imdb.images.data_mean);
    
    images = zeros(H, W, CH, 'single');
    for i=1:numel(cell_images)
        im = single(imresize(cell_images{i},[H,W]));
        images(:,:,:, i) = im - imdb.images.data_mean;
    end
    
    net.eval({'input', images});
    
    % obtain the CNN otuput
    scores(:,counter:counter2) = squeeze(gather(net.vars(net.getVarIndex('prob')).value));
    counter=counter2;
end
[~,pred_label]=max(scores,[],1);
acc=sum(pred_label==imdb.images.labels(imdb.images.set==setL))/size(pred_label,2);
