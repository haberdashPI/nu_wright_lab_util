data{
  int<lower=0> N; // num observations
  int<lower=1> K; // num of predictors
  matrix[N,K] x;
  vector[N] y;
}
parameters{
  vector[K] coefs;
  real<lower=0> sigma;
}
model{
  y ~ normal(x * coefs,sigma);
}
