function [layers, errors, params, timeSpent] = nnet(data, labels, params, layers)

%% [layers, errors, params, timeSpent] = nnet(data, labels, params, layers) trains a neural network using the set of parameters params.
%%
%% Inputs:
%% - data is the matrix containing the datapoints, one per row;
%% - labels is the matrix containing the labels, one per row;
%% - params is a structure containing the set of parameters for the network. Information on these parameters may be found in nnetDefaultParams.m .
%% - layers (optional) is a trained network. Use it to continue the optimization.
%%
%% Outputs:
%% - layers is a structure array containing all the parameters of the neural network.
%%			In particular layers(i).W and layers(i).B are the weight matrix and the bias between the i-th and the i-th + 1 layer;
%% - errors is the list of errors at the last iteration;
%% - params is the original structure with the missing entries filled;
%% - timeSpent is a vector containing the computation time for every iteration.
%%
%% Copyright Nicolas Le Roux, 2012. Released under the WTFPL.

% Extract the dimensions of the data.
[nSamples nValues] = size(data);

params.nSamples = nSamples;

% Fill the missing parameter fields, asking for user input if needed.
params = nnetDefaultParams(params);

% If the nTrain parameter was set to a value lower than 1, use it as a proportion.
if params.nTrain <= 1
    nTrain = round(params.nTrain * nSamples);
else
    nTrain = round(params.nTrain);
end
params.nTrain = nTrain;

% If the nValidation parameter was set to a value lower than 1, use it as a proportion.
if params.nValidation <= 1
    nValidation = round(params.nValidation * nSamples);
else
    nValidation = round(params.nValidation);
end
params.nValidation = nValidation;

batchSize = params.batchSize;

% Making sure there are enough datapoints in the matrix provided.
if nTrain + nValidation > nSamples
    error('Too few samples'); end

% Create batches.
% Batches for the validation set are not required, it is just to avoid memory issues.
train = createBatches(data(1:nTrain,:), labels(1:nTrain, :), batchSize);
validation = createBatches(data(nTrain + (1:nValidation),:), labels(nTrain + (1:nValidation), :), 1000);

nLabels =  cols(labels);
if nargin < 4
    % Initialize the layers of the neural network.
    [layers, gradient] = initializeLayers(params, nValues, nLabels);
else
    % Only initialize the gradient.
    [~, gradient] = initializeLayers(params, nValues, nLabels);
end

% Train the network.
[layers, errors, timeSpent] = nnetTrain(layers, gradient, params, train, validation);
