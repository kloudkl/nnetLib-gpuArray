function [ y ] = dropout( x, ratio )
if nargin < 2
    ratio = 0.5;
end
size_x = size(x);
mask = rand(size_x);
y = zeros(size_x);
index = mask > ratio;
y(index) = x(index);

end