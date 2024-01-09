

// stan code for sanabria behavioral econ

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 vector<lower=0>[Nr] y; //Local Rate stream
 int s[Nr]; //subject stream
 vector<lower=0>[Nr] Pr_req; //Programemd PR stream
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
real t_beta_11;
real w_beta_11;
real u_beta_11; 
real H_beta_11; 

real t_beta_18;
real w_beta_18;
real u_beta_18; 
real H_beta_18; 

//intercept_rme;
vector[Ns] t_int_rme; 
vector[Ns] w_int_rme;
vector[Ns] u_int_rme;
vector[Ns] H_int_rme; 

//lever height rme;
vector[Ns] t_beta_11_rme;
vector[Ns] w_beta_11_rme;
vector[Ns] u_beta_11_rme;
vector[Ns] H_beta_11_rme; 

vector[Ns] t_beta_18_rme;
vector[Ns] w_beta_18_rme;
vector[Ns] u_beta_18_rme;
vector[Ns] H_beta_18_rme;

vector<lower=0>[Ns] sigma; 
//real<lower=0> sigma; 

}

transformed parameters {

vector<lower=0>[Nr] row_Bn; 
vector<lower=0>[Nr] row_Pn; 
vector<lower=0>[Nr] row_sigma; 
vector<lower=0>[Nr] t;
vector<lower=0>[Nr] w;
vector<lower=0>[Nr] u;
vector<lower=0>[Nr] H; 
vector<lower=0,upper=1>[Nr] row_s; 

for (si in 1:Nr)
{
  if (LHStream[si] == 0)
  {
    t[si] = exp((t_int + t_int_rme[s[si]]));
    w[si] = exp((w_int + w_int_rme[s[si]]));
    u[si] = exp((u_int + u_int_rme[s[si]]));
    H[si] = exp((H_int + H_int_rme[s[si]]));
  }
  else if (LHStream[si] == 1)
  {
    t[si] = exp((t_int + t_int_rme[s[si]])+((t_beta_11 + t_beta_11_rme[s[si]])*1));
    w[si] = exp((w_int + w_int_rme[s[si]])+((w_beta_11 + w_beta_11_rme[s[si]])*1));
    u[si] = exp((u_int + u_int_rme[s[si]])+((u_beta_11 + u_beta_11_rme[s[si]])*1));
    H[si] = exp((H_int + H_int_rme[s[si]])+((H_beta_11 + H_beta_11_rme[s[si]])*1));
  }
  else if (LHStream[si] == 2)
  {
    t[si] = exp((t_int + t_int_rme[s[si]])+((t_beta_18 + t_beta_18_rme[s[si]])*1));
    w[si] = exp((w_int + w_int_rme[s[si]])+((w_beta_18 + w_beta_18_rme[s[si]])*1));
    u[si] = exp((u_int + u_int_rme[s[si]])+((u_beta_18 + u_beta_18_rme[s[si]])*1));
    H[si] = exp((H_int + H_int_rme[s[si]])+((H_beta_18 + H_beta_18_rme[s[si]])*1));
  }

  row_s[si] = H[si]/(1+H[si]);
  row_sigma[si] = sigma[s[si]];

  row_Pn[si] = u[si]*pow(((u[si]*RewMagStream[si])/(t[si]+Pr_req[si]/w[si])),row_s[si]/(1-row_s[si]));
  row_Bn[si] = Pr_req[si]/((t[si]+Pr_req[si]/w[si])*(1+(1/row_Pn[si])));
  

}



}

model {

//intercept priors
t_int ~ normal(0,3);
w_int ~ normal(0,3);
u_int ~ normal(0,3); 
H_int ~ normal(0,3); //approximately flat on s 
sigma ~ normal(0,1) T[0,]; //Gelman recommends 

//beta priors
t_beta_11 ~ normal(0,2);
w_beta_11 ~ normal(0,2);
u_beta_11 ~ normal(0,2); 
H_beta_11 ~ normal(0,2);

t_beta_18 ~ normal(0,2);
w_beta_18 ~ normal(0,2);
u_beta_18 ~ normal(0,2); 
H_beta_18 ~ normal(0,2);

//rme priors
t_int_rme ~ normal(0,1);
w_int_rme ~ normal(0,1);
u_int_rme ~ normal(0,1); 
H_int_rme ~ normal(0,1);

t_beta_11_rme ~ normal(0,1);
w_beta_11_rme ~ normal(0,1);
u_beta_11_rme ~ normal(0,1); 
H_beta_11_rme ~ normal(0,1);

t_beta_18_rme ~ normal(0,1);
w_beta_18_rme ~ normal(0,1);
u_beta_18_rme ~ normal(0,1); 
H_beta_18_rme ~ normal(0,1);

//likelihood
y ~ normal(row_Bn,row_sigma) T[0,]; 
//y ~ lognormal(row_Bn,row_sigma); 

}



