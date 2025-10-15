module Robustbase

using DataFrames

## Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: hbk, stackloss, wood, animals

greet() = print("Hello World from Robustbase!")

export greet

## Data
export hbk, stackloss, wood, animals

end
