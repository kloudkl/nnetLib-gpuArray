function [ output_args ] = gpu_data_layout_benchmark( input_args )
%GPU_DATA_LAYOUT_BENCHMARK Summary of this function goes here
%   Detailed explanation goes here
clear('all');
gpuDevice(1);

images = gpuArray.randn(256, 256, 3, 100);
filter = gpuArray.randn(7, 7);

tic;
for i = 1:10
    for imageIndex = 1:size(images, 4)
        for channel = 1:size(images, 3)
            conv2(images(:, :, channel, imageIndex), filter, 'valid');
        end
    end
end
toc


end

