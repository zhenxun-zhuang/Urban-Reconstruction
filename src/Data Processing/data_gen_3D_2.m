function data=data_gen_3D_2(num_points,noise_level,outlier_rate)

num_out = floor(num_points * outlier_rate);%number of outliers
num_in   = num_points - num_out;%number of inliers

outliers=2*rand(num_out,3)-1;

num_planes=3;
points_per_plane=round(sqrt(num_in/num_planes)*2);
l=(linspace(-1,1,points_per_plane))';%*2 in order to ensure there are enough inliers after ReLU
[x,y]=meshgrid(l,l);
x=reshape(x,points_per_plane*points_per_plane,1);
y=reshape(y,points_per_plane*points_per_plane,1);
z=zeros(size(y))-0.95;
points=[x,y,z;x,z,y;z,x,y];
%points=[x,y,z1];

points_n = points+noise_level*randn(size(points));%add gaussian noise
index = find(points_n(:,1)>-1 & points_n(:,1)<1 & points_n(:,2)>-1 & points_n(:,2)<1 & points_n(:,3)>-1 & points_n(:,3)<1);%values in [-1 1],[-1,1]
index_perm = randperm(size(index,1), num_in);%randomly permulate
index = index(index_perm);
inliers = [points_n(index,1), points_n(index,2), points_n(index,3)];

data=[inliers;outliers];
data=data(randperm(num_points),:);

scatter3(data(:,1), data(:,2), data(:,3));

data=[3,num_points,0;data];%dimension,numpoints;points_coordinations
csvwrite('data_3D_2.csv',data);