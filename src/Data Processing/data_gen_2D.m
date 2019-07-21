% Generate a 2D synthetic dataset consists of some lines and some outliers
% Input parameters:
%   num_points is the desired size of the dataset
%   noise_level is the Standard Deviation used in Additive White Gaussian Noise
%   outlier_rate is the percentage of outliers in the dataset
function data = data_gen_2D(num_points, noise_level, outlier_rate)

num_out = floor(num_points * outlier_rate); % Number of outliers
num_in = num_points - num_out; % Number of inliers

% Generate outliers by uniform randomization
outliers = 2*rand(num_out, 2) - 1;

% Generate inliers of four lines forming a square rotated by 45 degree
num_lines = 4; % Number of lines in the dataset
points_per_line = floor(num_in / num_lines); % Uniformly distribute samples to each line
x1 = (linspace(-1, 0, points_per_line*2))'; % Generate more samples to ensure enough inliers after truncation
y1 = x1*1 + 1;
x2 = (linspace(0, 1, points_per_line*2))';
y2 = x2*1 - 1;
x3 = (linspace(0, 1, points_per_line*2))';
y3 = -x3*1 + 1;
x4 = (linspace(-1, 0, points_per_line*2))';
y4 = -x4*1 - 1;
y = [x1, y1; x2, y2; x3, y3; x4, y4];
% Apply Additive White Gaussian Noise
y_n = y + noise_level*randn(size(y));
% Only take values in the range ([-1 1], [-1,1])
index = find((y_n(:, 1) > -1) & (y_n(:, 1) < 1) & (y_n(:, 2) > -1) & (y_n(:, 2) < 1));
% Randomly pick desired number of samples 
index_perm = randperm(size(index, 1), num_in);
index = index(index_perm);
inliers = [y_n(index, 1), y_n(index, 2)];
% Randomly permute all samples
data = [inliers; outliers];
data = data(randperm(num_points), :);
% Draw the dataset
scatter(data(:,1), data(:,2));
% Save the dataset as a CSV file in which the first line is [Dimension,
% Num_points] conforming with code specification
data=[2, num_points; data];
csvwrite('data.csv',data);