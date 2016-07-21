dataset_path='./../data/MIT_split/test/street'

contents = dir(sprintf('%s/*.jpg',dataset_path)); 
for i = 1:numel(contents)
      filename = contents(i).name;
      im= imread(sprintf('%s/%s',dataset_path,contents(i).name));
      [~,~,alexNet]=inference_alexnet(im, alexNet, 1);
      pause
end