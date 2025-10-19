module Robustbase

using DataFrames
using Statistics
using StatsBase: tiedrank

using LinearAlgebra
using Random
using Logging
using Plots

import Base: show
import StatsBase: CovarianceEstimator

using Distributions: sample, cdf, pdf, quantile, Normal, Chisq, Gamma

## Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: hbk, stackloss, wood, animals

greet() = print("Hello World from Robustbase!")

include("RobustScale.jl")
include("RobustCovariance.jl")

export greet,
    MAD_scale,
    Tau,
    Qn,
    Tau_location,
    Tau_scale,
    Qn_scale,
    CovClassic,
    CovMcd,
    CovOgk,
    DetMcd,
    fit!,
    scale,
    location,
    covariance,
    correlation,
    dd_plot


## Data
export hbk, stackloss, wood, animals

end
