module Robustbase

using DataFrames
using Statistics

import Base: show

using Distributions: sample, cdf, pdf, quantile, Normal, Chisq, Gamma

## Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: hbk, stackloss, wood, animals

greet() = print("Hello World from Robustbase!")

include("RobustScale.jl")

export greet,
    MAD_scale,
    Tau,
    Qn,
    Tau_location,
    Tau_scale,
    Qn_scale,
    fit!,
    scale,
    location


## Data
export hbk, stackloss, wood, animals

end
