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

# Numpy 
import numpy as np

def log_likelihood_gaussian(x,mu,sigma):
  """
    Compute log likelihood for Gaussian 

    Parameters:
    ----------- 
      x: (N,)
        Data points 
      mu: float
        Location parameter
      sigma: float
        Scale parameter

    Returns:
    -------
      logZ: float 
        Log likelihood
  
  """
  # Number of sample points
  N = len(x) 

  # Normalization factor
  t0 = -0.5*np.log(2*np.pi)

  # Term1: -log(sig)
  t1 = -np.log(sigma)

  # Term2: 
  t2  = (x-mu)**2
  t2  = -0.5*t2.sum()/sigma**2

  # Complete output
  logZ = N*t0 + N*t1 + t2

  return logZ

# ---------------------------------------------------------------------

def log_lognormal_likelihood(x,mu,sigma):
  """
    Power law

    Parameters:
    -----------
      x: (N,) 
        Data
      mu: float
        Location parameter 
      sigma: float
        Scale parameter
  """ 
  
  # Total number of data points
  N = len(x)
 
  # Term ind. of x 
  A = -.5*N*np.log(2*np.pi) - N*np.log(sigma)

  # Inverse half variance 1/(2*sigma^2)
  ivar = .5/sigma**2

  # Build vector of values
  Lx  = np.log(x)
  vec = -Lx - ivar*(Lx-mu)**2

  # Compute sum
  logL = A + vec.sum() 

  return logL

# ---------------------------------------------------------------------

def uprior_uniform(u,x1,x2):
  """
    Return sample from uniform prior with range [x1,x2)
  """
  return u*(x2-x1) + x1

# ---------------------------------------------------------------------

def uprior_jefferys(u,x1,x2):
  """
    Return sample from truncated Jeffery's prior:

    p(x)dx = A * dx/x,

    where

    A = 1/log(x2/x1),

    and p(x) = 0 outside  [x1,x2]. 

    Stats: 
    mean(x)   = (x2-x1)/log(x2/x1)
    median(x) = sqrt(x2*x1)

    Parameters:
    -----------
      u: (N,) 
        Uniform random variate in range
      x1: float
        Lower bound
      x2: float
        Upper bound

    Returns:
    --------
      x: (N,)
        Sample from truncated Jeffery's prior

  """
  return x1*np.exp(u*np.log(x2/x1))
  
