module Robustbase

## Predefined datasets used in outlier detection literature
include("data.jl")
import .DataSets: phones, hbk, stackloss
import .DataSets: weightloss, hs93randomdata, woodgravity
import .DataSets: hills, softdrinkdelivery, animals

greet() = print("Hello World from Robustbase!")

export greet

## Data
export hbk, stackloss, wood, animals

end
