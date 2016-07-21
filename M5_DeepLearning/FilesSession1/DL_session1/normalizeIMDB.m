function imdb = normalizeIMDB(imdb)

averagIm=imdb.images.data_mean;
averagIm=imresize(averagIm,[size(imdb.images.data,1),size(imdb.images.data,2)]);

data = imdb.images.data;

% zero mean
data = bsxfun(@minus, data, averagIm);

% expand to fill the range [-128, 128]
range_min = min(data(:));
range_max = max(data(:));
range_multiplier = 127./max(abs(range_min),range_max);
imdb.images.data = data .* range_multiplier;
end
