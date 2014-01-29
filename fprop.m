function layers = fprop(layers, data)

%% Forward propagation in a neural network.
%% The output of each layer is stored in layers(i).output

nLayers = length(layers);
nSamples = rows(data);

output = data;
for i = 1:nLayers
    W = layers(i).W;
    B = layers(i).B;
    func = layers(i).func;
    
    totalInput = output*W + repmat(B, nSamples, 1);
    switch func
        case 'none'
            output = totalInput;
        case 'sigm'
            output = sigm(totalInput);
        case 'tanh'
            output = tanh(totalInput);
        case 'nic'
            output = nic(totalInput);
        case 'softplus'
            output = softplus(totalInput);
        case 'relu'
            output = relu(totalInput);
        case 'dropout'
            output = dropout(totalInput);
        otherwise
            error('Unknown transfer function');
    end
%     if layers(i).dropout_ratio > 0
%         output = output .* layers(i).dropout_mask;
%     end
    layers(i).output = output;
    
end

% Adding the weight decay errors
for i = 1:nLayers
    wdValue = layers(i).wdValue;
    % Are we due for a weight decay computation now?
    if wdValue > 0
        k = numel(layers(i).W);
        switch layers(i).wdType
            case 0
                layers(i).wdCost = 0;
            case 1
                layers(i).wdCost = wdValue * sum(abs(layers(i).W(:))) / k;
            case 2
                layers(i).wdCost = .5 * wdValue * sum(layers(i).W(:).^2) / k;
            case 3                
                layers(i).wdCost = wdValue * (sum(abs(layers(i).W(:))) +...
                    .5 * sum(layers(i).W(:).^2)) / k;
        end
    else
        layers(i).wdCost = 0;
    end
end
