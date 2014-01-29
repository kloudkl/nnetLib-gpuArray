function [layers, gradient] = initializeLayers(params, sizeInput, sizeOutput)

%% [layers, gradient] = initializeLayers(params, sizeInput, sizeOutput) sets
%% all the fields in the neural networks (represented by the structure array
%% 'layers') and the gradient network using the sets of parameters stored in
%% params. The two arguments sizeInput and sizeOutput are here to determine the
%% sizes of the first and last layers.

% The list of sizes of each layer.
Nh = params.Nh;

nLayers = length(Nh) + 1;
layers = repmat(struct('W', zeros(0)), 1, nLayers);

% If there are no hidden layers, we have a linear classifier
if ~length(Nh)
    layers(1).size = [sizeInput, sizeOutput];
else
    layers(1).size = [sizeInput, Nh(1)];
    for i = 2:nLayers-1
        layers(i).size = [Nh(i-1), Nh(i)];
    end
    layers(nLayers).size = [Nh(end), sizeOutput];
end

% For each layer, initialize the parameters. If the number of parameters is less than the number of layers, all the subsequent layers have the last value of the parameters.
for i = 1:nLayers
    % For the last layer, if we use a 'nll' or 'ce' cost, the last transfer function must be linear.
    if i == nLayers && (strcmp(params.cost, 'nll') || strcmp(params.cost, 'ce'))
        layers(i).func = 'none';
    else
        layers(i).func = extractParam(params.func, i);
    end
    layers(i).updates = 0;
    layers(i).startLR = extractParam(params.startLR, i);
    layers(i).eps = 0;
    layers(i).decr = extractParam(params.decr, i);
    layers(i).adaptiveLR = extractParam(params.adaptiveLR, i);
    layers(i).use_adagrad = false;
    layers(i).use_adadec = false;
    if strcmp(layers(i).adaptiveLR, 'adagrad')
        layers(i).use_adagrad = true;
    elseif  strcmp(layers(i).adaptiveLR, 'adadec')
        layers(i).use_adadec = true;
    end
    layers(i).startAdaGrad = extractParam(params.startAdaGrad, i);
    layers(i).gammaAdaDec = extractParam(params.gammaAdaDec, i);
%     layers(i).taoAdaDec = extractParam(params.taoAdaDec, i);
    layers(i).initialMomentum = extractParam(params.initialMomentum, i);
    layers(i).initialMomentumIteration = extractParam(params.initialMomentumIteration, i);
    layers(i).finalMomentum = extractParam(params.finalMomentum, i);

    layers(i).W = gpuArray.randn(layers(i).size)/sqrt(layers(i).size(1));
    %     mask = gpuArray.rand(layers(i).size) < 0.1;
    %     layers(i).W(mask) = 0;
    layers(i).B = gpuArray.zeros(1, layers(i).size(2));
    layers(i).wdType = extractParam(params.wdType, i);
    layers(i).wdValue = extractParam(params.wdValue, i);
    layers(i).wdCost = gpuArray.zeros(1, layers(i).size(2));
    layers(i).dropout_ratio = 0.5;
    m = gpuArray.rand(layers(i).size) < layers(i).dropout_ratio;
    layers(i).dropout_mask = gpuArray.ones(layers(i).size);
    layers(i).dropout_mask(m) = 0;
    gradient(i).W = gpuArray.zeros(layers(i).size);
    gradient(i).B = gpuArray.zeros(1, layers(i).size(2));
end % for i = 1:nLayers

end % function [layers, gradient] = initializeLayers(params, sizeInput, sizeOutput)

function output = extractParam(input, i)
    output = input(min(i, length(input)));
    if iscell(output)
        output = output{1};
    end
end
