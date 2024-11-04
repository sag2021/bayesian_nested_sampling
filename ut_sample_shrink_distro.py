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
# Unit test: sample_shrink_distro
#
# Checks the mean/std versus the known analytic values
#
# If t is drawn from the shrink distribution, then 
#
# mean(log(t)) = -1/n
# std(log(t))  = 1/n
#

# Standard modules
import numpy as np
import matplotlib.pyplot as plt

# RNG
from numpy.random import default_rng

# Local
from nested_sample import sample_shrink_distro

# Number of samples
N_samples = int(1e4)

# Number of objects
n = 13

# Setup RNG
seed = 31415 
rng  = default_rng(seed)

# Sample
t     = sample_shrink_distro(n,N_samples,rng)
log_t = np.log(t)

# Compute stats.
log_t_mean = log_t.mean()
log_t_std  = log_t.std()

# Print Results
print ("Results:")
print ("Samples:    ",N_samples)
print ("Objects:    ",n        )
print ("Mean [num]: ",log_t_mean)
print ("Mean [ana]: ",-1./n     )
print ("Std: [num]: ",log_t_std )
print ("Std: [ana]: ",1./n )

# Draw log-log histogram
bins = np.logspace(log_t.min(),0,64)
plt.hist(t,bins=bins,density=True)
plt.xscale('log')
plt.yscale('log')
plt.show()
print ("Done")
