function [acc]=eval_acc(imdb,net)

% run the CNN
setL=3;  % 3 is the test set

net.eval({'input', imdb.images.data(:,:,:,(imdb.images.set==setL))});
net.mode='test';

%netstats=extractStatsFn(net)
% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value;
scores = squeeze(gather(scores));
[~,pred_label]=max(scores,[],1);
acc=sum(pred_label==imdb.images.labels(imdb.images.set==setL))/size(pred_label,2);