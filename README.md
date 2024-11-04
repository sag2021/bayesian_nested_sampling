# Overview
Implementation of the Bayesian nested sampling method [^1][^2]. Computes the Bayesian evidence and information for a parametric model.
Also computes estimates of the parameters and uncertainties based on a moments of the posterior. 

Based on the approach described by Sivia,D.S., Skilling, J. (2006) [^1]. Implements Monte-Carlo error estimation.

Ongoing project. 

[^1]: Sivia,D.S., Skilling, J.: 2006,Data Analysis - A Bayesian Tutorial (2nd ed.),Oxford Science Publications Oxford University Press, ISBN:  978-0198568322 
[^2]: John Skilling. "Nested sampling for general Bayesian computation." Bayesian Anal. 1 (4) 833 - 859, December 2006. https://doi.org/10.1214/06-BA127 

# Example

Example code finds the location and scale parameter for a log-normal distribution using a log-normal sample. 

