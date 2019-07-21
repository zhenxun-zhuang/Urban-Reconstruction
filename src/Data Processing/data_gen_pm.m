function data_gen_pm(outlier_level)

load('data_proto_2D.mat');

data_0 = trans(data, [0.7, 0.7], 55, [0, 0], 0.003,0.5);
data_1 = trans(data, [0.2, 0.3], 45, [0.2, 0.2], 0.003, 0.2);
data_2 = trans(data, [0.3, 0.1], 85, [-0.6, 0.5], 0.003, 0.3);
data_3 = trans(data, [0.2, 0.2], 115, [-0.5, -0.3], 0.003, 0.2);
%data_4 = trans(data, [0.4, 0.1], 175, [0.5, -0.7], 0.002, 0.3);
%data_5 = trans(data, [0.1, 0.4], 225, [0.5, 0.7], 0.003, 0.3);
%data_6 = trans(data, [0.2, 0.3], 275, [-0.5, -0.75], 0.002, 0.2);
inliers = [data_0; data_1; data_2; data_3];

index = find((inliers(:, 1) > -1) & (inliers(:, 1) < 1) & (inliers(:, 2) > -1) & (inliers(:, 2) < 1));%values in [-1 1],[-1,1]
inliers = [inliers(index, 1), inliers(index, 2)];

num_inliers = size(inliers, 1);

num_outliers = round(num_inliers * outlier_level / (1 - outlier_level));
outliers = 2*rand(num_outliers, 2) - 1;

data = [inliers; outliers];
num_total = size(data, 1);

data = data(randperm(num_total), :);
scatter(data(:, 1),data(:, 2));

data = [2, num_total; data];
csvwrite('data_pm_2D.csv', data);

end

function data_new = trans(data, s, r, t, noise, downsample)

m_s = [s(1), 0, 0; 0, s(2), 0; 0, 0, 1];
m_r = [cosd(r), -sind(r), 0;  sind(r), cosd(r), 0; 0, 0, 1];
m_t = [1, 0, t(1); 0, 1, t(2); 0, 0, 1];

data = data';
num = size(data, 2);
data = [data; ones(1, num)];

data_new = m_t * m_r * m_s * data;
data_new = data_new./data_new(3, :);
data_new = data_new(1:2, :);

data_new = data_new + noise*randn(size(data_new));

num_new = round(num * downsample);
data_new = data_new(:,  randperm(num, num_new));
data_new = data_new';

end