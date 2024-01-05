

// stan code for sanabria behavioral econ

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 vector<lower=0>[Nr] y; //Local Rate stream
 int s[Nr]; //subject stream
 vector<lower=0>[Nr] Pr_req; //AltPR stream
 vector<lower=0>[Nr] LHStream; //Lever height stream
 vector<lower=1>[Nr] RewMagStream; //Reward Magnitude Stream
}

parameters {

//intercept
real t_int; 
real w_int;
real u_int; 
real H_int; 

//lever height effect
real t_beta;
real w_beta;
real u_beta; 
real H_beta; 

//intercept_rme;
vector[Ns] t_int_rme; 
vector[Ns] w_int_rme;
vector[Ns] u_int_rme;
vector[Ns] H_int_rme; 

//lever height rme;
vector[Ns] t_beta_rme;
vector[Ns] w_beta_rme;
vector[Ns] u_beta_rme;
vector[Ns] H_beta_rme; 

vector<lower=0>[Ns] sigma; 

}

transformed parameters {

vector<lower=0>[Nr] row_Bn; 
vector<lower=0>[Nr] row_sigma; 
vector<lower=0>[Nr] row_Pn; 
vector<lower=0>[Nr] row_common_denom; 
vector<lower=0,upper=1>[Nr] row_s; 
vector[Nr] t;
vector[Nr] w;
vector[Nr] u;
vector[Nr] H; 

for (si in 1:Nr)
{
  t[si] = (t_int + t_int_rme[s[si]]) + ((t_beta + t_beta_rme[s[si]])*LHStream[si]);
  w[si] = (w_int + w_int_rme[s[si]]) + ((w_beta + w_beta_rme[s[si]])*LHStream[si]);
  u[si] = (u_int + u_int_rme[s[si]]) + ((u_beta + u_beta_rme[s[si]])*LHStream[si]);
  H[si] = (H_int + H_int_rme[s[si]]) + ((H_beta + H_beta_rme[s[si]])*LHStream[si]);

  row_common_denom[si] = exp(t[s[si]]) + (Pr_req[si]/exp(w[s[si]]));
  row_s[si] = exp(H[s[si]])/(1+exp(H[s[si]]));
  row_Pn[si] = exp(u[s[si]])*pow(((exp(u[s[si]])*RewMagStream[si])/row_common_denom[si]),(row_s[si]/(1-row_s[si])));
  row_Bn[si] = Pr_req[si]/(row_common_denom[si]*(1+(1/row_Pn[si])));
  
  row_sigma[si] = sigma[s[si]];

}

}

model {

//intercept priors
t_int ~ normal(0.16,5);
w_int ~ normal(1.05,5);
u_int ~ normal(2.95,5); 
H_int ~ normal(1,5);
sigma ~ inv_gamma(1,1); 

//beta priors
t_beta ~ normal(0,2);
w_beta ~ normal(0,2);
u_beta ~ normal(0,2); 
H_beta ~ normal(0,2);

//rme priors
t_int_rme ~ normal(0,2);
w_int_rme ~ normal(0,2);
u_int_rme ~ normal(0,2); 
H_int_rme ~ normal(0,2);

t_beta_rme ~ normal(0,2);
w_beta_rme ~ normal(0,2);
u_beta_rme ~ normal(0,2); 
H_beta_rme ~ normal(0,2);

//likelihood
y ~ normal(row_Bn,row_sigma) T[0,];

}



