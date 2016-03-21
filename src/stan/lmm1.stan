data {
  int<lower=1> n; // num individuals
  int<lower=1> g_1; // num type 1 groups 

  int<lower=0> k; // num fixed predictors 
  int<lower=1> h_1; // num type 1 ind predictors

  int<lower=1> l_1; // num type 1 group predictors
  
  int<lower=1,upper=g_1> gg_1[n];  // group for individual
  matrix[n,k] A; // fixed predictors
  matrix[n,h_1] B_1; // individual predictors type 1
  matrix[g_1,l_1] G_1; // type 1 group predictors

  real y[n]; // outcomes

  real fixed_mean_prior;

  real group1_mean_prior;
  real group1_var_prior;
  real group1_cor_prior;

  real prediction_error_prior;
}
parameters {
  // fixed coefficitents
  vector[k] alpha; 

  // underlying parameters for optimized error coefficients

  // group type 1
  matrix[h_1,g_1] z_1; 
  cholesky_factor_corr[h_1] L_Omega_1;
  vector<lower=0>[h_1] tau_1; // prior scale
  matrix[l_1,h_1] gamma_1; // group scale

  // prediction error scale
  real<lower=0> sigma; 
}
transformed parameters {
  // error coefficients
  matrix[g_1,h_1] beta_1;

  // predicted outcomes
  beta_1 <- G_1 * gamma_1 + (diag_pre_multiply(tau_1,L_Omega_1) * z_1)';
}
model {
  // distribution of error coefficients
  to_vector(z_1) ~ normal(0,1);
  to_vector(gamma_1) ~ normal(0,group1_mean_prior);
  tau_1 ~ cauchy(0,group1_var_prior);
  L_Omega_1 ~ lkj_corr_cholesky(group1_cor_prior);

  // distribution of fixed coefficients
  alpha ~ normal(0,fixed_mean_prior);

  // distribution of outcomes
  sigma ~ normal(0,prediction_error_prior);
  for(i in 1:n){
    real y_hat;
    y_hat <- A[i]*alpha + B_1[i]*beta_1[gg_1[i]]';
    y[i] ~ normal(y_hat,sigma);
  }
}
generated quantities {
  matrix[h_1,h_1] Sigma_beta_1;

  Sigma_beta_1 <- diag_pre_multiply(tau_1,L_Omega_1) *
      diag_pre_multiply(tau_1,L_Omega_1)';
}
