function layers = nnetUpdate(layers, gradient)

%% nnetUpdate update the parameters of the neural network defined by layers.
%% gradient is a structure the same size as layers containing all gradients.

persistent sum_squared_gradient;

nLayers = length(layers);

if isempty(sum_squared_gradient)
    sum_squared_gradient = repmat(struct('W', gpuArray.zeros(0), ...
        'B', gpuArray.zeros(0)), 1, nLayers);
end

for i = nLayers:-1:1
    layers(i).updates = layers(i).updates + 1;
    %     eps = layers(i).startLR; % constant learning rate scheduling
    %     eps = layers(i).startLR * exp(-layers(i).updates / 10); % exponential
    eps = layers(i).startLR / (1 + layers(i).decr * layers(i).updates); % power
    
    if layers(i).use_adagrad || layers(i).use_adadec
        if i == nLayers && layers(i).updates == 1
            existsOnGPU(sum_squared_gradient(i).W)
        end
        if isempty(sum_squared_gradient(i).W)
            sum_squared_gradient(i).W = gradient(i).W .^ 2;
        end
        if isempty(sum_squared_gradient(i).B)
            sum_squared_gradient(i).B = gradient(i).B .^ 2;
        end
    end
    if layers(i).use_adagrad
        sum_squared_gradient(i).W =  sum_squared_gradient(i).W + ...
            gradient(i).W .^ 2;
        sum_squared_gradient(i).B =  sum_squared_gradient(i).B + ...
            gradient(i).B .^ 2;
    end
    if layers(i).use_adadec
        sum_squared_gradient(i).W =  layers(i).gammaAdaDec * sum_squared_gradient(i).W + ...
            gradient(i).W .^ 2;
        sum_squared_gradient(i).B =  layers(i).gammaAdaDec * sum_squared_gradient(i).B + ...
            gradient(i).B .^ 2;
    end
    
    if layers(i).use_adagrad && layers(i).updates >= layers(i).startAdaGrad
        if layers(i).updates == layers(i).startAdaGrad
            eps = eps * 3;
        end
        layers(i).W = layers(i).W - eps * (gradient(i).W ./ sqrt(1 + sum_squared_gradient(i).W));
        layers(i).B = layers(i).B - eps * (gradient(i).B ./ sqrt(1 + sum_squared_gradient(i).B));
    else
        % It seems like creating extra variables W and B is faster. Aaaah, the mysteries of MATLAB.
        %         gradW = gradient(i).W;
        %         gradB = gradient(i).B;%
        %         W = layers(i).W;
        %         B = layers(i).B;
        %         W = W - eps*gradW;
        %         B = B - eps*gradB;
        %         layers(i).W = W;
        %         layers(i).B = B;
        
        layers(i).W = layers(i).W - eps * gradient(i).W;
        layers(i).B = layers(i).B - eps * gradient(i).B;
    end
    
    layers(i).eps = eps;
end
