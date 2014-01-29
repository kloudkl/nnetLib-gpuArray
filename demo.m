% This file demonstrates how to use the library by training a neural network on a small version of the MNIST database.
clear('all');
gpuDevice(1);

% 1 - Loads the mnist dataset. If it is not available, print where to download it.
try
    load('mnist.mat');
catch
    fprintf('You need to download the MNIST dataset for this demo to work.\nIt is available at http://nicolas.le-roux.name/code/mnist_small.mat\n');
    return;
end

% 2 - Creates a default set of parameters.
parameters;
% params.Nh = [200 200 200]; % Use only one hidden layer for the demo.
% params.nIter = 10; % Only do 10 passes through the data for the demo.
params.save = 0; % Do not save the network on disk for the demo.


% 3 - Split the dataset into train+valid and test.

num_training_samples = min(34000, size(data, 1));
data_train = data(1 : num_training_samples, :);
labels_train = labels(1 : num_training_samples, :);
shuffle = randperm(num_training_samples);
data_train = data_train(shuffle, :);
labels_train = labels_train(shuffle, :);

num_test_samples = min(30000, size(test_data, 1));
data_test = test_data(1 : num_test_samples, :);
labels_test = test_labels(1 : num_test_samples, :);
num_test_samples = size(data_test, 1);
shuffle = randperm(num_test_samples);
test_data = test_data(shuffle, :);
test_labels = test_labels(shuffle, :);

% 4 - Train the neural network.
labels_train = transform_labels(labels_train, params.cost); 
[layers, errors, params, timeSpent] = nnet(data_train, labels_train, params);

% 5 - Test the network using the classification error.
labels_test = transform_labels(labels_test, params.cost); 
[predicted, errors] = nnetTest(data_test, labels_test, 'class', layers);

% 6 - Display the test error.
fprintf('Test classification error rate: %g%%\n', 100*mean(errors));