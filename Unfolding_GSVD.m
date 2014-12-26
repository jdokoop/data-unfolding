%{
    Code demonstrating the use of GENERALIZED singular value decomposition 
    with Tikhonov regularization for solving discrete data unfolding 
    (i.e., deconvolution) problems.

    We simulate an exponentially falling spectrum, distorted with a 
    Gaussian convolution kernel with random white noise. The signal is
    reconstructed imposing minimization requirements on the curvature, as
    well as boundary conditions on the solution.

    AUTHOR: J. Orjuela-Koop
    DATA: December 2014
%}

%Define general parameters of the simulation
lim_inf = 0.1;
lim_sup = 5;
DIM = 100;

plot_picard = 0;
plot_signals = 1;
plot_basis = 0;
plot_crossvalidation = 0;
generate_cross_validation = 1;

aux = linspace(lim_inf,lim_sup,DIM);
[Y_coord,X_coord] = meshgrid(aux,aux);

%Define the truth distribution and Gaussian convolution kernel
f_truth = @(x) exp(-x);
truth_orig = exp(-aux);
f_kern = @(x) 0.05*exp(-x.^2);

width = (1./sqrt(X_coord*X_coord+Y_coord*Y_coord))+0.02;
K = ((lim_sup-lim_inf)/DIM)*(exp(-(X_coord-Y_coord).^2./(2*width)));      %Response matrix

%Generate data
truth = exp(-aux);
b_exact = K*truth';
e = 0.01*max(b_exact)*randn(DIM,1);              %Generate Gaussian noise
b_measured = b_exact + e;                        %Add noise

%Minimize curvature by defining discrete second derivative operator
first_row_L = [-2 1 zeros(1,DIM-2)];
L = toeplitz(first_row_L); 

%Impose reflective boundary conditions
L(1,1) = -1;
L(end,end) = -1;

%Generalized Singular Value Decomposition
[U,V,W,C,S] = gsvd(K,L);
X = inv(W');
sm = zeros(DIM,2);
sm(:,1) = diag(C);
sm(:,2) = diag(S);
lambda = 9.5;
fi = (sm(:,1).^2) ./ (sm(:,1).^2 + lambda^2*sm(:,2).^2);

%Solve system using GSVD
x_reg = zeros(DIM,1);

for i=1:DIM
    x_reg = x_reg + (fi(i)/sm(i,1))*dot(U(:,i)',b_measured)*X(:,i);
end

%Generate the generalized Picard plot
svd_comps = abs(U'*b_measured);                           %SVD coefficients
svd_comps_ratio = svd_comps ./ sm(:,1);                   %SVD coeffs ratio
svd_comps_ratio_filtered = fi .* svd_comps ./ sm(:,1);    %SVD coeffs ratio with smoothing

%Cross-validation curve to determine optimal smoothing parameter
g_lambda = zeros (1,70);
lambda_vals = zeros (1,70);
x_lambda = zeros(70,DIM);
lambda_0 = 0.002;
dLambda = 0.08;
x_reg2 = zeros(DIM,1);
    
for i=1:500
    lambda_vals(i) = i*dLambda+lambda_0;
    fi2 = (sm(:,1).^2) ./ (sm(:,1).^2 + (i*dLambda+lambda_0)^2*sm(:,2).^2);
    
    for j=1:DIM
       x_reg2 = x_reg2 + (fi2(j)/sm(j,1))*dot(U(:,j)',b_measured)*X(:,j);
    end
    
    x_lambda(i,:) = x_reg2;
    g_lambda(i) = (norm(K*x_reg2 - truth')^2)/(DIM-sum(fi2))^2;
    x_reg2 = zeros(DIM,1);
end

%Plot validation curve
%Optimal value of lambda minimizes the curve
figure;
plot(lambda_vals,g_lambda,'-r','LineWidth',2);
title('Validation Curve G(\lambda)')
xlabel('\lambda')
ylabel('G(\lambda)')

%Plot family of solutions for different values of lambda
figure;
mesh(x_lambda,'FaceColor','blue','EdgeColor','none');
camlight right;
lighting phong;
title('Family of Solutions')
xlabel('p_{T} [a.u.]')
ylabel('\lambda')

%Generalized Picard plot
figure;
semilogy(flip(svd_comps),'r.');                               
hold on;
semilogy(flip(svd_comps_ratio),'g.');
hold on;
semilogy(flip(svd_comps_ratio_filtered),'b.');
title('Picard Plot')
legend('u_i^T\cdot b','u_i^T\cdot b/\sigma_i','\phi_i u_i^T\cdot b/\sigma_i')
xlabel('i')

%Plot reconstructed signal
figure;
plot(aux,b_measured,'b-','LineWidth',2);
hold on;
plot(aux,x_reg,'bo');
hold on;
plot(aux,truth_orig,'-r','LineWidth',0.5);
title('Spectrum Reconstruction with GSVD')
legend('Measured','Reconstructed','Theoretical')
xlabel('p_{T} [a.u.]')