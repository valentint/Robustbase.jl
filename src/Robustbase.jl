module Robustbase

using DataFrames
using Statistics
using StatsBase: tiedrank, cov2cor

using LinearAlgebra
using Random
using Logging
using StatsPlots

import Base: show
import StatsBase: CovarianceEstimator

using Distributions: sample, cdf, pdf, quantile, Normal, Chisq, Gamma

## Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: hbk, stackloss, wood, animals

##  greet() = print("Hello World from Robustbase!")

include("RobustScale.jl")
include("RobustCovariance.jl")
include("plots.jl")

export ## greet,
    MAD_scale,
    Tau,
    Qn,
    Tau_location,
    Tau_scale,
    Qn_scale,
    CovClassic,
    RobustCovariance,
    CovMcd,
    CovOgk,
    DetMcd,
    fit!,
    scale,
    location,
    covariance,
    correlation,
    distance,
    dd_plot,
    qq_plot,
    distance_plot,
    tolellipse_plot


## Data
export hbk, stackloss, wood, animals

end
