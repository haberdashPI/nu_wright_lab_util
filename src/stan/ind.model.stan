data {
  int<lower=0> N; // num individuals
  int<lower=1> K; // num ind predictors
  int<lower=1> J; // num groups
  int<lower=1> L; // num group predictors
  int<lower=1,upper=J> jj[N];  // group for individual
  matrix[N,K] x; // individual predictors
  matrix[J,L] u; // group predictors
  vector[N] y; // outcomes

  real group_mean_prior;
  real group_var_prior;
  real group_cor_prior;
  real prediction_error_prior;
}
parameters {
  matrix[K,J] z; 
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0>[K] tau; // prior scale
  matrix[L,K] gamma; // group scale
  real<lower=0> sigma; // prediction error scale
}
transformed parameters {
  matrix[J,K] beta;
  vector[N] y_hat;
  beta <- u * gamma + (diag_pre_multiply(tau,L_Omega) * z)';
  for (n in 1:N) y_hat[n] <- x[n] * beta[jj[n]]';
}
model {
  to_vector(z) ~ normal(0,1);
  to_vector(gamma) ~ normal(0,group_mean_prior);
  tau ~ cauchy(0,group_var_prior);
  L_Omega ~ lkj_corr_cholesky(group_cor_prior);

  sigma ~ normal(0,prediction_error_prior);
  y ~ normal(y_hat, sigma);
}
generated quantities{
  matrix[K,K] Sigma_beta;

  Sigma_beta <- diag_pre_multiply(tau,L_Omega) * diag_pre_multiply(tau,L_Omega)';
}
