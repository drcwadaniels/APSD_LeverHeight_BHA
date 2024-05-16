

// stan code for sanabria behavioral econ

data {

 int<lower=1> Nr; // number of data points
 int<lower=1> Ns; // number of subjects
 vector[Nr] y; //Local Rate stream
 int s[Nr]; //subject stream
 vector<lower=0>[Nr] Pr_req; //Programemd PR stream
 vector<lower=0>[Nr] LHStream; //Lever height stream
 vector<lower=1>[Nr] RewMagStream; //Reward Magnitude Stream
 vector<lower=0>[Nr] SessionTime; //TotalSessionTime
 real minTau; //min Tau
}

parameters {

//intercept
real t_int; 
real w_int;
real z_int; 
real Qlogit_int; 
real sigma_int;

//lever height effect
real t_beta_11;
real w_beta_11;
real z_beta_11; 
real Qlogit_beta_11; 
real sigma_beta_11;

real t_beta_18;
real w_beta_18;
real z_beta_18; 
real Qlogit_beta_18; 
real sigma_beta_18;

//intercept_rme;
vector[Ns] t_int_rme; 
vector[Ns] w_int_rme;
vector[Ns] z_int_rme;
vector[Ns] Qlogit_int_rme; 
vector[Ns] sigma_int_rme;

//lever height rme;
vector[Ns] t_beta_11_rme;
vector[Ns] w_beta_11_rme;
vector[Ns] z_beta_11_rme;
vector[Ns] Qlogit_beta_11_rme; 
vector[Ns] sigma_beta_11_rme;

vector[Ns] t_beta_18_rme;
vector[Ns] w_beta_18_rme;
vector[Ns] z_beta_18_rme;
vector[Ns] Qlogit_beta_18_rme;
vector[Ns] sigma_beta_18_rme;

}

transformed parameters {

vector[Nr] row_Bn; 
vector<lower=0>[Nr] row_Qplus;
vector<lower=0>[Nr] row_upper; 
vector<lower=0>[Nr] row_rho; 
vector<lower=0>[Nr] row_demand;
vector<lower=0>[Nr] row_sigma; 
vector<lower=0>[Nr] t;
vector<lower=0>[Nr] w;
vector<lower=0>[Nr] z;
vector<lower=0,upper=1>[Nr] Qprop; 

for (si in 1:Nr)
{
  if (LHStream[si] == 0)
  {
    t[si] = exp((t_int + t_int_rme[s[si]]))+minTau;
    w[si] = exp((w_int + w_int_rme[s[si]]));
    z[si] = exp((z_int + z_int_rme[s[si]]));
    Qprop[si] = exp(Qlogit_int + Qlogit_int_rme[s[si]])/(1+exp(Qlogit_int + Qlogit_int_rme[s[si]])); 
    row_sigma[si] = exp((sigma_int + sigma_int_rme[s[si]]));

  }
  else if (LHStream[si] == 1)
  {
    t[si] = exp((t_int + t_int_rme[s[si]])+((t_beta_11 + t_beta_11_rme[s[si]])*1))+minTau;
    w[si] = exp((w_int + w_int_rme[s[si]])+((w_beta_11 + w_beta_11_rme[s[si]])*1));
    z[si] = exp((z_int + z_int_rme[s[si]])+((z_beta_11 + z_beta_11_rme[s[si]])*1));
    Qprop[si] = exp((Qlogit_int + Qlogit_int_rme[s[si]])+((Qlogit_beta_11 + Qlogit_beta_11_rme[s[si]])*1))/(1+exp((Qlogit_int + Qlogit_int_rme[s[si]])+((Qlogit_beta_11 + Qlogit_beta_11_rme[s[si]])*1)));
    row_sigma[si] = exp((sigma_int + sigma_int_rme[s[si]])+((sigma_beta_11 + sigma_beta_11_rme[s[si]])*1));


  }
  else if (LHStream[si] == 2)
  {
    t[si] = exp((t_int + t_int_rme[s[si]])+((t_beta_11 + t_beta_11_rme[s[si]])*1)+((t_beta_18 + t_beta_18_rme[s[si]])*1))+minTau;
    w[si] = exp((w_int + w_int_rme[s[si]])+((w_beta_11 + w_beta_11_rme[s[si]])*1)+((w_beta_18 + w_beta_18_rme[s[si]])*1));
    z[si] = exp((z_int + z_int_rme[s[si]])+((z_beta_11 + z_beta_11_rme[s[si]])*1)+((z_beta_18 + z_beta_18_rme[s[si]])*1));
    Qprop[si] = exp((Qlogit_int + Qlogit_int_rme[s[si]])+((Qlogit_beta_11 + Qlogit_beta_11_rme[s[si]])*1)+((Qlogit_beta_18 + Qlogit_beta_18_rme[s[si]])*1))/(1+exp((Qlogit_int + Qlogit_int_rme[s[si]])+((Qlogit_beta_11 + Qlogit_beta_11_rme[s[si]])*1)+((Qlogit_beta_18 + Qlogit_beta_18_rme[s[si]])*1)));
    row_sigma[si] = exp((sigma_int + sigma_int_rme[s[si]])+((sigma_beta_11 + sigma_beta_11_rme[s[si]])*1)+((sigma_beta_18 + sigma_beta_18_rme[s[si]])*1));

  }

  row_rho[si] = t[si]+(Pr_req[si]/(w[si]*RewMagStream[si]));
  
  row_upper[si] = (z[si]*pow(1/t[si],2)+1)/(z[si]*(1/t[si])+t[si]);
  
  row_Qplus[si] =  row_upper[si] * Qprop[si]; 
  
  row_demand[si] = row_Qplus[si]*((z[si]+t[si]*row_rho[si])/(z[si]+pow(row_rho[si],2))); 

  row_Bn[si] = ((row_demand[si]*Pr_req[si])/(RewMagStream[si]*SessionTime[si]))/60;

}



}

model {

//intercept priors
t_int ~ normal(0,10);
w_int ~ normal(0,10);
z_int ~ normal(0,10); 
Qlogit_int ~ normal(0,10);
sigma_int ~ normal(-2,0.25);

//beta priors
t_beta_11 ~ normal(0,1);
w_beta_11 ~ normal(0,1);
z_beta_11 ~ normal(0,1); 
Qlogit_beta_11 ~ normal(0,1);
sigma_beta_11 ~ normal(0,0.25);

t_beta_18 ~ normal(0,1);
w_beta_18 ~ normal(0,1);
z_beta_18 ~ normal(0,1); 
Qlogit_beta_18 ~ normal(0,1);
sigma_beta_18 ~normal(0,0.25);

//rme priors
t_int_rme ~ normal(0,1);
w_int_rme ~ normal(0,1);
z_int_rme ~ normal(0,1); 
Qlogit_int_rme ~ normal(0,1);
sigma_int_rme ~ normal(0,0.25);


t_beta_11_rme ~ normal(0,1);
w_beta_11_rme ~ normal(0,1);
z_beta_11_rme ~ normal(0,1); 
Qlogit_beta_11_rme ~ normal(0,1);
sigma_beta_11_rme ~ normal(0,0.25);

t_beta_18_rme ~ normal(0,1);
w_beta_18_rme ~ normal(0,1);
z_beta_18_rme ~ normal(0,1); 
Qlogit_beta_18_rme ~ normal(0,1);
sigma_beta_18_rme ~ normal(0,0.25);

//likelihood
y ~ normal(log(row_Bn),row_sigma); //sigma

}



