%{
    Code demonstrating the use of singular value decomposition with 
    Tikhonov regularization for solving discrete data unfolding 
    (i.e., deconvolution) problems.

    We simulate an exponentially falling spectrum, distorted with a 
    Gaussian convolution kernel with random white noise. The signal is
    reconstructed using all of the kernel's left singular vectors, with
    a filtering factor depending on an "arbitrary" smoothing scalar.

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

%Tikhonov smoothing
lambda = 0.096;
F_num = D*D;
F_denom = D*D + lambda*lambda*eye(DIM);
F = F_num / F_denom;
ff = diag(F);                                    %Filter factors

%Generate the Picard plot
svd_comps = abs(U'*b_measured);                  %SVD coefficients
svd_comps_ratio = svd_comps ./ d;                %SVD coeffs ratio

%Solve the system using Tikhhonov regularization
x = zeros(1,DIM);
for i=1:DIM
    x = x + ff(i)*dot(U(:,i)',b_measured)/d(i) * V(:,i)';
end

%Plot filter factors
figure;
semilogy(ff,'bo');
title('Filter Factors')
xlabel('i')
ylabel('\phi_i')

%Generate Picard plot
figure;  
semilogy(d,'b.');                               
hold on;
semilogy(svd_comps,'r.');
hold on;
semilogy(svd_comps_ratio,'g.');
title('Picard Plot')
legend('\sigma_i','u_i^T\cdot b','u_i^T\cdot b/\sigma_i')
xlabel('i')

%Plot reconstructed signal
figure;
plot(aux,b_measured,'b-','LineWidth',2);
hold on;
plot(aux,x,'ob');
hold on;
plot(aux,truth_orig,'-r','LineWidth',0.5);
title('Spectrum Reconstruction with Tikhonov Regularization')
legend('Measured','Reconstructed','Theoretical')
xlabel('p_{T} [a.u.]')