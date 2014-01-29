function [errors, gradient] = computeCost(layers, labels, cost)

output = layers(end).output;
[nSamples nLabels] = size(output);
nLayers = length(layers);

% Compute the cost due to weight decays.
if isfield(layers(1), 'wdCost')
	wdCost = sum([layers.wdCost]);
else
	wdCost = 0;
end

switch cost
case 'mse'
	diff = output - labels;
	errors = .5*sum(diff.^2,2) + wdCost;
	gradient = diff/nSamples;
case 'ce'
	errors = -sum(output.*labels - log(1 + exp(output)), 2) + wdCost;

	% Avoid overflows.
	errors(output > 20) = -sum(output(output > 20).*labels(output > 20) - output(output > 20), 2) + wdCost;
	gradient = -(labels - sigm(output))/nSamples;

case 'nll'
	softmaxOutput = softmax(output, 2);
	errors = sum(-log(softmaxOutput).*labels,2) + wdCost;
	gradient = (- labels + softmaxOutput)/nSamples;
case 'class'
	if size(output, 2) == 1
		errors = ( sign(output) ~= (2*labels-1) );
	else
		[~, valueOutput] = max(output, [], 2);
		errors = ( valueOutput ~= labels*(1:nLabels)' );
		if nargout > 1, error('This cost is not designed for training'); end
	end
end
