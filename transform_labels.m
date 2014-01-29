function [ new_labels ] = transform_labels( labels, cost_type )
%TRANSFORM_LABELS Summary of this function goes here
%   Detailed explanation goes here

new_labels = labels;

% If we are using negative log-likelihood with only one label, create a one-hot encoding.
if strcmp(cost_type, 'nll') && cols(labels) == 1
    new_labels = oneHot(labels);
end

% If the cost is the cross-entropy, make sure the labels are 0 and 1.
if strcmp(cost_type, 'ce') && cols(labels) == 1
    if unique(labels) == [-1; 1], new_labels = (labels+1)/2 ;
    elseif unique(labels) == [0; 1],
    else error('Wrong setting of the labels.');
    end
end

end

