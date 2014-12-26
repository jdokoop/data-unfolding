%{
    Code demonstrating the use of truncated singular value decomposition
    for solving discrete data unfolding (i.e., deconvolution) problems.

    We simulate an exponentially falling spectrum, distorted with a 
    Gaussian convolution kernel with random white noise. The signal is
    reconstructed using a subset of the convolution kernel's left singular
    vectors. 

    AUTHOR: J. Orjuela-Koop
    DATA: December 2014
%}

%Define general parameters of the simulation
lim_inf = 0.1;
lim_sup = 5;
DIM = 100;

aux = linspace(lim_inf,lim_sup,DIM);
[Y,X] = meshgrid(aux,aux);

%Define the truth distribution and Gaussian convolution kernel
f_truth = @(x) exp(-x);
truth_orig = exp(-aux);
f_kern = @(x) 0.05*exp(-x.^2);

width = (1./sqrt(X*X+Y*Y))+0.02;

K = ((lim_sup-lim_inf)/DIM)*(exp(-(X-Y).^2./(2*width)));      %Response matrix

%Generate data
truth = exp(-aux);
b_exact = K*truth';
e = 0.01*max(b_exact)*randn(DIM,1);              %Generate Gaussian noise
b_measured = b_exact + e;                        %Add noise

%Singular Value Decomposition
[U,D,V] = svd(K);
d = diag(D);

%Generate the Picard plot
svd_comps = abs(U'*b_measured);                  %SVD coefficients
svd_comps_ratio = svd_comps ./ d;                %SVD coeffs ratio

%Solve the system using TSVD regularization
r = 16;                                           %Keep only 'r' sing. vals.
x = zeros(1,DIM);
for i=1:r
    x = x + dot(U(:,i)',b_measured)/d(i) * V(:,i)';
end

%Generate Picard Plot
figure;
semilogy(d,'b.');                               
hold on;
semilogy(svd_comps,'r.');
hold on;
semilogy(svd_comps_ratio,'g.');
title('Picard Plot')
legend('\sigma_i','u_i^T\cdot b','u_i^T\cdot b/\sigma_i')
xlabel('i')

%Plot some left-singular vectors
figure;
plot(V(:,1),'r-','LineWidth',2);
hold on;
plot(V(:,2),'k-','LineWidth',2);
hold on;
plot(V(:,3),'b-','LineWidth',2);
hold on;
plot(V(:,50),'g-','LineWidth',1);
title('Selected Left-Singular Vectors')
legend('v_1','v_2','v_3','v_{50}')

%Plot TSVD reconstruction
figure;
plot(aux,b_measured,'b-','LineWidth',2);
hold on;
plot(aux,x,'ob');
hold on;
plot(aux,truth_orig,'-r','LineWidth',0.5);
title('TSVD Spectrum Reconstruction')
legend('Measured','Reconstructed','Theoretical')
xlabel('p_{T} [a.u.]')

%Plot response matrix
figure;
imagesc(K);
title('Detector Response Matrix')

