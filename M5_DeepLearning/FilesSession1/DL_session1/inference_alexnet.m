function [bestScore,best,alexNet]=inference_alexnet(im, alexNet, display_flag)

if nargin<3
    display_flag=0;
end
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, alexNet.meta.normalization.imageSize(1:2)); % rescale [227 227 3]
im_ = im_ - alexNet.meta.normalization.averageImage;

% run the CNN
alexNet.eval({'input', im_});

% obtain the CNN otuput
scores = alexNet.vars(alexNet.getVarIndex('prob')).value;
scores = squeeze(gather(scores));

% show the classification results
if(display_flag)
    [bestScore, best] = max(scores);
    figure(1) ; clf ; imagesc(im);
    title(sprintf('%s (%d), score %.3f', alexNet.meta.classes.description{best}, best, bestScore));
end
