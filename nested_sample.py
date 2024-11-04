# BSD 3-Clause License
# 
# Copyright (c) 2024, S.A. Gilchrist
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Import modules
import sys
import copy 
import numpy as np 
from time import time as timer

# Local
from nested_base import nested_base_class

#  Conversion to Bits
NAT_TO_BITS = 1./np.log(2.)

# ----------------------------------------------------------------------------------------

class model_uniform(nested_base_class):
  """
    Data model with "uniform" prior. In this context, "uniform"
    is an prior that can be sampled by a transformation of a uniform
    variate.
  """

  def __init__(self,n,flogL,upriors,hyper,rng,step=.1,uprior_samples=32):
    """

      Parameters:
      -----------
        n: int
          Number of parameters
        flogL: function
          Function that returns the log likelihood. Has the form flogL(data,*params),
          where *params are the parameters of the model (self.params)
        upriors: list-like
          List of functions that map a uniform variate to a sample from the prior
        hyper: list-like 
          List of hyperparameters lists. Each element is itself a list of hyperparameters passed
          to the uprior function for the ith parameter 
        rng: 
          Random number generator
        step: float
          Step size for prior exploration
        uprior_samples: int
          Number of samples used to explore the prior

    """

    # Values Check
    if(len(hyper) != n):
      raise ValueError("Length of hyperparameter list must match number of parameters")
    if(len(upriors) != n): 
      raise ValueError("Number of priors must match number of parameters")
    if(uprior_samples < 1):
      raise ValueError("uprior_samples must be at least one")

    # Random number generator
    self.rng = rng

    # Model parameters
    self.params  = np.zeros(n)
    self.nparams = n
    self.nudevs  = self.nparams

    # Step size and number of samples for prior
    self.step            = step
    self.uprior_samples  = uprior_samples

    # Prior function and hyperparameters
    self.upriors = upriors
    self.hyper   = hyper
    self.pmap    = None

    # Likelihood function
    self.flogL = flogL

    # Values: log(L), Log(Wt)
    self.logL  = 0.
    self.logWt = 0.

  def prior(self,data):
    """
      Initialize parameters and logL
    """
  
    # Generate uniform variates
    self.u = self.rng.uniform(0,1,self.nparams)

    # Map from hypercube to prior space
    for i in range(self.nparams):
      self.params[i] = self.upriors[i](self.u[i],*self.hyper[i])

    # Evaluate log likelihood
    self.logL = self.logLhood(data,self.params)

  def logLhood(self,data,params):
    """
      Evaluate log Likelihood      
    """
    return self.flogL(data,*params)
  
  def explore(self,data,logLstar):
    """
      Explore prior space within the Likelihood constraint

      Parameters:
      -----------
        data:
          Passed to logLhood
        logLstar: float

    """
    # Construct new object
    new_obj = copy.deepcopy(self)

    # Initialize accept/reject counters
    accept = 0
    reject = 0

    # Initialize step
    step         = new_obj.step
    params_trial = np.zeros(new_obj.nparams)

    # 
    # Extrapolation loop
    #
    for m in range(self.uprior_samples):

      # Generate new points in hypercube
      u_trial  = new_obj.u + step*new_obj.rng.uniform(-1,+1,new_obj.nparams)
      u_trial -= np.floor(u_trial)

      # Map from hypercube to parameter space
      for i in range(new_obj.nparams):
        params_trial[i] = new_obj.upriors[i](u_trial[i],*new_obj.hyper[i])
        
      # Compute log(L)
      logL_trial = new_obj.logLhood(data,params_trial)

      # If step is accepted, then update object values
      if(logL_trial > logLstar):
        new_obj.u      = u_trial
        new_obj.params = params_trial
        new_obj.logL   = logL_trial 
        accept += 1
      else: 
        reject += 1

      # Refine step 
      if( accept > reject ):
        step *= np.exp(1.0 / accept);
      if( accept < reject ):
        step /= np.exp(1.0 / reject);

    return new_obj

  def __getitem__(self,name):
    i = int(name)
    return self.params[i]

# ---------------------------------------------------------------------
  
def sample_shrink_distro(n,size,rng):
  """
    Sample shrinkage distribution.

    p(t) = n * t^(n-1) 

    Equation (9.9) from Sivia,D.S., Skilling, J.: 2006, 
    Data Analysis - A Bayesian Tutorial (2nd ed.),
    Oxford Science Publications Oxford University Press, ISBN:  978-0198568322 

    Parameters:
    ----------- 
      n: int 
        Number of nested sampling objects
      size: tuple
        The shape of the sample array
      rng: func
        Numpy random number generator

    Returns:
    --------
      t: 
        Sample from distribution
  """

  # Uniform variates in the range [0,1)
  u = rng.uniform(size=size)

  # Output
  inv_n = 1./n
  t     = (1-u)**inv_n

  return t

# ---------------------------------------------------------------------

def new_evidence(logZ_old,logWt_worst):
  """
    Update the discrete evidence sum:

        ___
        \
    Z = /__ A_k 
         k
 
    A_k = h_k * L_k.


    Equation (9.13) from Sivia,D.S., Skilling, J.: 2006, 
    Data Analysis - A Bayesian Tutorial (2nd ed.),
    Oxford Science Publications Oxford University Press, ISBN:  978-0198568322 

    Equation (13) from Skilling, J.: 2006, 
    "Nested sampling for general Bayesian computation", 
    Bayesian Analysis, 1, Number 4,p833
    DOI: 10.1214/06-BA127

    Parameters:
    -----------
      logZ_old: (N,)
        Log of current evidence
      logWt_worst: (N,)
        Log of the weight: log(Wt) = log(h) + log(L), where h is the width

    Returns:
    --------
      res: float
        Updated log(Z) 

  """ 
  return np.logaddexp(logZ_old,logWt_worst)

# ---------------------------------------------------------------------

def new_information(H_old,logWt_worst,logL_worst,logZ_old,logZ_new):
  """
    Update information:
        ___
        \
    H = /__ (A_k/Z) * log(L_k/Z)
         k

    A_k: h_k(t)*L_k, where h_k is the width
    L_k: likelihood at nested sample k
    Z:   Overall evidence

    Because Z depends on A_k and L_k at every previous sample, the
    expression for the H update is somewhat complicated

    Parameters:
    -----------
      H: (N,)
        Current information value
      logWt_worst: (N,)
        Log Weights for worst object
      logL: (N,)
        Log likelihood of worst object
      logZ_old: (N,)
        Log evidence for current iteration
      logZ_new:
        Log evidence for new iteration

    Returns:
    --------
      H_new: float
        Updated information
 
  """
  H_new = (np.exp(logWt_worst - logZ_new)*logL_worst 
        +  np.exp(logZ_old - logZ_new)*(H_old + logZ_old) - logZ_new)

  return H_new

# ---------------------------------------------------------------------

def quatify(post_samples,logZ,name):
  """
    Compute value of variable from posterior samples. 

    Based on Equation (38)-(41) of  
    Skilling, J.: 2006, "Nested sampling for general Bayesian computation", 
    Bayesian Analysis, 1, Number 4,p833
    DOI: 10.1214/06-BA127
 
    Parameters:
    -----------
      post_samples:  
        List of posterior samples
      logZ: (n,)
        Log of evidence for each sample
      name: str
        Name of parameter. At present, this should just be a number (as a string)

    Returns:
    --------
      mu_Q: float
        Mean of parameter over the posterior. This serves as an estimate
        of the parameter. 
      sigma_Q: float
        STD of parameter over posterior. This measures the width of the posterior. 
      mu_Q_err: float
        Monte-Carlo error due to the shrink distro. Serves as an estimate of the 
        numerical Monte-Carlo error. At minimum, this must be less than sigam_Q for the estimate
        of mu_Q to be taken seriously 
      mu_Q_err_std: 
        Monte-Carlo error in sigma_Q 

  """

  # Check length of post_samples
  npost_samples = len(post_samples)
  if(npost_samples == 0):
    raise ValueError("post_samples must have at least one element")
 
  # Arrays to hold estimates as functions of the shrink factor t 
  mu_Qt  = np.zeros(len(logZ))
  var_Qt = np.zeros(len(logZ))

  # Array to hold posterior weights
  post_wt = np.zeros([npost_samples,len(logZ)])   
 
  # Compute probability weights. Note that the weights are function 
  # of the shrink factor t: w = w(t)
  for i,obj in enumerate(post_samples):
    post_wt[i,:] = np.exp(obj.logWt - logZ)

  # Compute mean over posterior
  for i,obj in enumerate(post_samples):
    _Qi     = obj[name] 
    mu_Qt  += post_wt[i,:]*_Qi

  # Compute variance over posterior
  for i,obj in enumerate(post_samples):
    _Qi      = obj[name]
    var_Qt  += post_wt[i,:]*(_Qi - mu_Qt)**2
    
  # Compute square root
  sigma_Qt = np.sqrt(var_Qt)

  # Compute mean over shrink factor t
  mu_Q    = mu_Qt.mean() 
  sigma_Q = sigma_Qt.mean() 

  # Compute STD over shrink factor 
  mu_Q_err    = mu_Qt.std()
  sigma_Q_err = sigma_Qt.std()
  
  return mu_Q,sigma_Q,mu_Q_err,sigma_Q_err

# ---------------------------------------------------------------------

def nested_samples(obj0,nobjs,data,nmax,rng,nsamples=64,dtype=np.float64):
  """
     Wrapper for nested_sample_core. Generates the object list from a 
     template class

    Parameters:
    -----------  
      nclass: class
        Class that defines the prior space and computes the likelihood 
        This class should be an extension of the nest_base abstract base class
      nobjs: int
        Number of objects to use during calculation
      data: List
        List of arguments passed to object methods
      nmax: int
        Max number of nested sampling iterations
      rng: function
        Numpy random number generator
      nsamples: int
        The number of Monte-Carlo samples for the shrink factor

    Returns:
    --------
      out: dict

  """

  objs = [obj0]

  # Generate objects
  for i in range(1,nobjs):
    objs.append(copy.deepcopy(obj0))

  # Set prior objects
  for i in range(nobjs):
    objs[i].prior(data)

  # Perform nested sampling
  out = nested_sample_core(objs,data,nmax,rng,nsamples,dtype=dtype)

  return out

# ---------------------------------------------------------------------

def initialize_nested_objects(obj0,nobjs,data,nmax,rng,nsamples=64,dtype=np.float64):
  """
    Initilize objects based on an initial template
  """

  objs = [obj0]

  # Generate objects
  for i in range(1,nobjs):
    objs.append(copy.deepcopy(obj0))

  # Set prior objects
  for i in range(nobjs):
    objs[i].prior(data)

  return objs
  
# ---------------------------------------------------------------------

def nested_sample_core(objs,args,nmax,rng,nsamples=64,n0=0,dtype=np.float64):
  """
    Core method for performing nested sampling. For this method the objects must already be created and 
    initialized. 

    Parameters:
    -----------
      objs: (nobjs,)
        List of objects. Should be an extension of the nest_base ABC
      args: List
        List of arguments passed to object methods
      nmax: int
        Max number of nested sampling iterations
      rng: function
        Numpy random number generator
      nsamples: int
        The number of Monte-Carlo samples for the shrink factor

    Returns:
    --------
      out: dict
        logZ_mean: float
          The mean of the log of the evidence over Monte-Carlo samples
        logZ_std: float
          The STD of the log of the evidence 
        H_mean: float 
          The Shannon information averaged over Monte-Carlo samples
        H_std: float
          The STD of the Shannon information over the Monte-Carlo samples

  """ 

  # Record start time
  tstart = timer()
 
  # Get number of objects
  nobjs = len(objs)

  # Check values
  if(nsamples < 1):
    raise ValueError("Nsamples must be greater than 1")
  if(nmax < 1):
    raise ValueError("Nmax must be greater than 1")
  if(nobjs < 1):
    raise ValueError("objs must have len > 0")
  
  # Initialize
  out          = {}                                 # Output dict
  post_samples = []                                 # List to contain posterior samples
  min_val      = np.finfo(dtype).min                # Smallest value for dtype
  H            = np.zeros(nsamples)                 # Information [nat units]
  logZ         = np.full([nsamples],min_val)        # Log of evidence
  logL         = np.zeros(nobjs)                    # Log of likelihood

  # Compute: 1/n
  inv_n = 1./nobjs

  # Compute width of first interval
  #
  if(nsamples > 1):
    _t       = sample_shrink_distro(nobjs,nsamples,rng)
    logwidth = np.log(1 - _t)
  else:
    logwidth = np.log(1 - np.exp(-inv_n))*np.ones(nsamples)

  # 
  # Main loop 
  #
  for nest in range(n0,n0+nmax):

    # For the first iteration: populate the vector with logL
    # For other iterations, only the object with index_worst has been changed so update
    # the vector with new value of logL
    if(nest==n0):
      for i in range(nobjs):
        logL[i] = objs[i].logL
    else:
      logL[index_worst] = objs[index_worst].logL
  
    # Find index with worst (smallest) likelihood
    index_worst = np.argmin(logL)
 
    # Compute weight log(A) = log(h) + log(L)
    objs[index_worst].logWt = logwidth + objs[index_worst].logL

    # Update evidence Z and information H
    logZnew = new_evidence(logZ,objs[index_worst].logWt)
    H       = new_information(H,objs[index_worst].logWt,objs[index_worst].logL,logZ,logZnew)
    logZ    = logZnew

    # Save for computing posterior samples
    post_samples.append(copy.deepcopy(objs[index_worst]))

    # Kill worst object in favour of copy of different survivor
    index_copy = index_worst
    while((index_copy == index_worst) and (nobjs>1)):
      index_copy = rng.integers(0,nobjs)
    logLstar = objs[index_worst].logL

    # Evolve copied object with constraint
    # Replaces worst object with index_copy 
    objs[index_worst] = objs[index_copy].explore(args,logLstar)
 
    # Shrink interval
    if(nsamples > 1):
      _t        = sample_shrink_distro(nobjs,nsamples,rng)
      logwidth += np.log(_t)
    else:
      logwidth -= inv_n

  # Record end time
  tend = timer()

  # Compute wall time [s]
  wtime = tend - tstart

  # Compute means
  out["H_mean"]      = H.mean()
  out["logZ_mean"]   = logZ.mean()

  # Compute rough estimate errors
  out["logZ_err_approx"] = np.sqrt(out["H_mean"]/len(objs))

  # Compute Monte-Carlo errors
  if(nsamples > 1):
    out["logZ_std"]  = logZ.std()
    out["H_std"]     = H.std()
  else:
    out["logZ_std"]  = 0
    out["H_std"]     = 0

  # Output
  out["logZ"]         = logZ
  out["H"]            = H
  out["nest"]         = nest
  out["nobjs"]        = nobjs
  out["post_samples"] = post_samples
  out["nsamples"]     = nsamples
  out["wtime"]        = wtime
  out["info"]         = info_string(out)
 
  return out

# ---------------------------------------------------------------------

def info_string(out):
  """
    Generate info. string
  """
    
  s1  = "Number of objects:   {:d} \n".format(out["nobjs"])
  s1 += "Number of iterations:{:d} \n".format(out["nest"])
  s1 += "Number of samples:   {:d} \n".format(out["nsamples"])
  s1 += "Wall time [s]:       {:g}   ".format(out["wtime"])
  
  return s1
