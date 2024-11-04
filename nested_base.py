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
# Base class for nested sampling object

# Imports
from abc import ABC,abstractmethod

class nested_base_class(ABC):
  """
    Abstract base class for nested sampling object
  """
 
  def __init__(self,rng,args):
    self.logL  = 0
    self.logWt = 0
    self.rng   = rng
  
  @abstractmethod
  def prior(self,args):
    """
      Initialize data by sampling the prior
    """
    pass

  @abstractmethod
  def explore(self,args,logLstar):
    """
      Explore prior
    """
    return 

  def get(self,name):
    """ 
      Get attribute value

      Parameters:
      -----------
        name: str
          Name of attribute 
    """
    return getattr(self,name)

  def __getitem__(self,key):
    """
      Access attribute like a dict
    """
    return self.get(str(key))

