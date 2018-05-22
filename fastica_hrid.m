clear all;
clc;
close all;

im_dimension=[32,64];
SizeofFeaturevector=im_dimension(1)*im_dimension(2)

matrix=[];
srcFiles = dir('F:\DATASETS\nordland_railway\training_final\*.png');  
for j = 1 : length(srcFiles)
filename = strcat('F:\DATASETS\nordland_railway\training_final\',srcFiles(j).name);
    I = imread(filename);
  
      gray=rgb2gray(I);
      gray_resize=imresize(gray,(im_dimension));
      
      feature_vector=gray_resize(:);
      
      matrix_cat=horzcat(matrix,feature_vector);
      matrix=matrix_cat;
 
end
    matrix_winter=matrix;
    
t=matrix_winter';

[row, col]=size(t);
%zero centering
mu=mean(t);
x=double(t)-repmat(mu,[row,1]);
j=row;
m=j;

%whitening
sx=cov(x);
sx=sx+0.6*eye(2048,2048);
[v,d]=eig(sx);
x=v*sqrtm(inv(d))*v'*x';
x=real(x);

[rows, cols]=size(x);

n=rows;
ncomp=2048;
W=zeros(rows,ncomp);

%itertive algorithm
maxiter=1000;
matrix=[];

for p=1:ncomp

    wp=rand(n,1);
    wp=wp/norm(wp);
    
for iter=1:maxiter
  u=wp'*x;
%   g=tanh(wp'*x);
%   dg=1-tanh(wp'*x).^2;
wp_old=wp;
g=u.^3;  %non-liear function
dg=3*u.^2;

   wp=(x*g')/m -dg*ones(m,1)*wp/m;
   wp=wp-W*W'*wp;
   wp=wp/norm(wp);

  if abs(abs(wp'*wp_old)-1)<0.0001      
            W(:,p)=wp_old;                 
             break; 
         end      
end


ica_number = sprintf('ICA number =  %d  (%d steps) ', p, iter);
   disp(ica_number)

end
final=W;
U=W'*double(t');
A=pinv(W');


% Reconstruction using independent components
 f=A*U;
  ncomp=2048;
  f=A(:,1:ncomp)*U(1:ncomp,:);

figure;
k=500;
for i=1:9
recon=reshape(uint8(f(:,k)),im_dimension);
subplot(3,3,i);
imshow(recon)
k=k+1;
end

