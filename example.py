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

# 
# Example. Fits log-normal distro to a log-normal sample.
#

# Standard modules
import numpy as np

# RNG
from numpy.random import default_rng

# Local
from nested_sample import *
from distros import log_lognormal_likelihood,uprior_uniform,uprior_jefferys

# Wrapper for likelihood
def log_likelihood(x,mu,sigma):
  return log_lognormal_likelihood(x,mu,sigma)

# Conversion to dB
DB = 10./np.log(10.)

# RNG
rng = default_rng(2021)

# Test data population parameters
MU    = 3.
SIGMA = 1.
N     = 128

# Prior range 
DELTA_MU  = +3
DELTA_SIG = +4

# Nested sampling parameters 
N_OBJECTS = 100
N_SAMPLES = 10000

print ("Number of data points: ",N)

# Generate sample data
data = rng.lognormal(MU,SIGMA,size=N)

# Setup tuples of priors and hyperparameters. The model has 
# two parameters (mu,sigma). Priors and hyper-parameters are needed
# for both. The order in upriors and hyper must match the order
# expected by the likelihood function.
#
# Prior for mean: Truncated uniform in range [-DELTA_MU,+DELTA_MU]
# Prior for sigma: Truncated Jeffery's in range [1e-DELTA_SIG,1e+DELTA_SIG]
#
#
upriors = (uprior_uniform,uprior_jefferys)
hyper   = ([-DELTA_MU,+DELTA_MU],[10.**-DELTA_SIG,10.**+DELTA_SIG])

# Construct model object
model = model_uniform(2,log_likelihood,upriors,hyper,rng)

# Perform nested sampling
res = nested_samples(model,N_OBJECTS,data,N_SAMPLES,rng)

# Print 
print("Population mean: {:g}".format(MU))
print("Population std:  {:g}".format(SIGMA))

# Print evidence/information
print("")
print ("LogZ    {:g} +/- {:g}         ".format(res["logZ_mean"]         ,res["logZ_std"])       )
print ("LogZ    {:g} +/- {:g}dB       ".format(res["logZ_mean"]*DB      ,res["logZ_std"]*DB)       )
print ("LogZ    {:g} +/- {:g} [approx]".format(res["logZ_mean"]         ,res["logZ_err_approx"]))
print ("H[bits] {:g} +/- {:g}         ".format(res["H_mean"]*NAT_TO_BITS,res["H_std"]*NAT_TO_BITS)     )

# Parameters
mu_x,sigma_x,mu_x_err,sigma_x_err = quatify(res["post_samples"],res["logZ"],"0")
mu_y,sigma_y,mu_y_err,sigma_y_err = quatify(res["post_samples"],res["logZ"],"1")

# Print x,y values with errors
print ("\nVariables over post")
print ("First error: variable STD")
print ("Second error: Monte-Carlo numerical")
print ("x: {:g} +/- {:g} +/- {:g}        ".format(mu_x,sigma_x,mu_x_err))
print ("y: {:g} +/- {:g} +/- {:g}        ".format(mu_y,sigma_y,mu_y_err))
print ("Done")

