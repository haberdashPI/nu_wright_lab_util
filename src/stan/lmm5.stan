data {
  int<lower=1> n; // num individuals
  int<lower=1> g_1; // num type 1 groups
  int<lower=1> g_2; // num type 2 groups
  int<lower=1> g_3; // num type 3 groups
  int<lower=1> g_4; // num type 4 groups
  int<lower=1> g_5; // num type 5 groups

  int<lower=0> k; // num fixed predictors
  int<lower=1> h_1; // num type 1 ind predictors
  int<lower=1> h_2; // num type 2 ind predictors
  int<lower=1> h_3; // num type 3 ind predictors
  int<lower=1> h_4; // num type 4 ind predictors
  int<lower=1> h_5; // num type 5 ind predictors

  int<lower=1> l_1; // num type 1 group predictors

  int<lower=1,upper=g_1> gg_1[n];  // group type 1 for individual
  int<lower=1,upper=g_2> gg_2[n];  // group type 2 for individual
  int<lower=1,upper=g_3> gg_3[n];  // group type 3 for individual
  int<lower=1,upper=g_4> gg_4[n];  // group type 4 for individual
  int<lower=1,upper=g_5> gg_5[n];  // group type 5 for individual

  matrix[n,k] A; // fixed predictors
  matrix[n,h_1] B_1; // individual predictors type 1
  matrix[n,h_2] B_2; // individual predictors type 2
  matrix[n,h_3] B_3; // individual predictors type 3
  matrix[n,h_4] B_4; // individual predictors type 4
  matrix[n,h_5] B_5; // individual predictors type 5

  matrix[g_1,l_1] G_1; // type 1 group predictors

  real y[n]; // outcomes

  real fixed_mean_prior;

  real group1_mean_prior;
  real group1_var_prior;
  real group1_cor_prior;

  real group2_var_prior;
  real group2_cor_prior;

  real group3_var_prior;
  real group3_cor_prior;

  real group4_var_prior;
  real group4_cor_prior;

  real group5_var_prior;
  real group5_cor_prior;

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

  // group type 2
  matrix[h_2,g_2] z_2;
  cholesky_factor_corr[h_2] L_Omega_2;
  vector<lower=0>[h_2] tau_2; // prior scale

  // group type 3
  matrix[h_3,g_3] z_3;
  cholesky_factor_corr[h_3] L_Omega_3;
  vector<lower=0>[h_3] tau_3; // prior scale

  // group type 4
  matrix[h_4,g_4] z_4;
  cholesky_factor_corr[h_4] L_Omega_4;
  vector<lower=0>[h_4] tau_4; // prior scale

  // group type 5
  matrix[h_5,g_5] z_5;
  cholesky_factor_corr[h_5] L_Omega_5;
  vector<lower=0>[h_5] tau_5; // prior scale

  // prediction error scale
  real<lower=0> sigma;
}
transformed parameters {
  // error coefficients
  matrix[g_1,h_1] beta_1;
  matrix[g_2,h_2] beta_2;
  matrix[g_3,h_3] beta_3;
  matrix[g_4,h_4] beta_4;
  matrix[g_5,h_5] beta_5;

  // predicted outcomes
  beta_1 <- G_1 * gamma_1 + (diag_pre_multiply(tau_1,L_Omega_1) * z_1)';
  beta_2 <- (diag_pre_multiply(tau_2,L_Omega_2) * z_2)';
  beta_3 <- (diag_pre_multiply(tau_3,L_Omega_3) * z_3)';
  beta_4 <- (diag_pre_multiply(tau_4,L_Omega_4) * z_4)';
  beta_5 <- (diag_pre_multiply(tau_5,L_Omega_5) * z_5)';
}
model {
  // distribution of error coefficients
  to_vector(z_1) ~ normal(0,1);
  to_vector(gamma_1) ~ normal(0,group1_mean_prior);
  tau_1 ~ cauchy(0,group1_var_prior);
  L_Omega_1 ~ lkj_corr_cholesky(group1_cor_prior);

  to_vector(z_2) ~ normal(0,1);
  tau_2 ~ cauchy(0,group2_var_prior);
  L_Omega_2 ~ lkj_corr_cholesky(group2_cor_prior);

  to_vector(z_3) ~ normal(0,1);
  tau_3 ~ cauchy(0,group3_var_prior);
  L_Omega_3 ~ lkj_corr_cholesky(group3_cor_prior);

  to_vector(z_4) ~ normal(0,1);
  tau_4 ~ cauchy(0,group4_var_prior);
  L_Omega_4 ~ lkj_corr_cholesky(group4_cor_prior);

  to_vector(z_5) ~ normal(0,1);
  tau_5 ~ cauchy(0,group5_var_prior);
  L_Omega_5 ~ lkj_corr_cholesky(group5_cor_prior);

  // distribution of fixed coefficients
  alpha ~ normal(0,fixed_mean_prior);

  // distribution of outcomes
  sigma ~ normal(0,prediction_error_prior);
  for(i in 1:n){
    real y_hat;
    y_hat <- A[i]*alpha + B_1[i]*beta_1[gg_1[i]]' + B_2[i]*beta_2[gg_2[i]]' +
      B_3[i]*beta_3[gg_3[i]]' + B_4[i]*beta_4[gg_4[i]]' + B_5[i]*beta_5[gg_5[i]]';
    y[i] ~ normal(y_hat,sigma);
  }
}
generated quantities {
  matrix[h_1,h_1] Sigma_beta_1;
  matrix[h_2,h_2] Sigma_beta_2;
  matrix[h_3,h_3] Sigma_beta_3;
  matrix[h_4,h_4] Sigma_beta_4;
  matrix[h_5,h_5] Sigma_beta_5;

  Sigma_beta_1 <- diag_pre_multiply(tau_1,L_Omega_1) *
      diag_pre_multiply(tau_1,L_Omega_1)';
  Sigma_beta_2 <- diag_pre_multiply(tau_2,L_Omega_2) *
      diag_pre_multiply(tau_2,L_Omega_2)';
  Sigma_beta_3 <- diag_pre_multiply(tau_3,L_Omega_3) *
      diag_pre_multiply(tau_3,L_Omega_3)';
  Sigma_beta_4 <- diag_pre_multiply(tau_4,L_Omega_4) *
      diag_pre_multiply(tau_4,L_Omega_4)';
  Sigma_beta_5 <- diag_pre_multiply(tau_5,L_Omega_5) *
      diag_pre_multiply(tau_5,L_Omega_5)';
}
