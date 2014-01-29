function preprocess_mnist
%PREPROCESS_MNIST Summary of this function goes here
%   Detailed explanation goes here
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');


image_mean = mean(images, 2);
image_std = std(images, 0, 2);
image_std(image_std == 0) = 1;

images = bsxfun(@rdivide, bsxfun(@minus, images, image_mean), image_std);
% images = whiten(images);
data = images';

test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

test_images = bsxfun(@rdivide, bsxfun(@minus, test_images, image_mean), image_std);
% test_images = whiten(test_images);
test_data = test_images';

save('mnist', 'image_mean', 'image_std', 'data', 'labels', 'test_data', ...
    'test_labels');

end

function whitened = whiten(x, type)
if nargin < 2
    type = 'pca';
end

% x num_pixels * num_samples
num_pixels = size(x, 1);
num_samples = size(x, 2);
sigma = x * x' / num_pixels;
[U, S, V] = svd(sigma);
xRot = U' * x;
epsilon = 0.1;
xPCAWhite = diag(1 ./ sqrt(diag(S) + epsilon)) * U' * x;
if strcmp(type, 'zca')
    whitened = U * xPCAWhite;
else
    whitened = xPCAWhite;
end

end

function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end


function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end