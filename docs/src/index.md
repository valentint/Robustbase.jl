# Robustbase.jl

Documentation for Robustbase.jl

# Univariate: Robust scales
The univariate module contains several univariate location and scale estimators. 
They all implement the abstract base class `RobustScale`, which has
the properties `location` and `scale` and can be fitted using the `fit!` method. Each 
class is expected to implement a `_calculate` method where the attributes `scale_` and
`location_` are set.

## Tau scale
```@docs
Robustbase.Tau
```

## Qn scale
```@docs
Robustbase.Qn
```
## Location
```@docs
Robustbase.location
```

## Scale
```@docs
Robustbase.scale
```

# Multivariate: Covariance
Various robust estimators of covariance matrices ("scatter matrices") have been
proposed in the literature, with different properties. The covariance module implements
several frequently used scatter estimators. They all use the new base class `RobustCovariance`
which builds on the `CovarianceEstimator` class in `StatsBase`.

## Classical Location and Scatter Estimation: CovClassic
```@docs
Robustbase.CovClassic
```

## Covariance matrix: covariance
```@docs
Robustbase.covariance
```

## Correlation matrix: correlation
```@docs
Robustbase.correlation
```

## Robust Location and Scatter Estimation via MCD: CovMcd
```@docs
Robustbase.CovMcd
```

## Deterministic MCD estimator: DetMCD
```@docs
Robustbase.DetMcd
```

## Orthogonalized Gnanadesikan-Kettenring estimator: CovOgk
```@docs
Robustbase.CovOgk
```

# Data sets
`Robustbase` includes several datasets that are often used in the robustness literature.
These datasets serve as standard examples and benchmarks, allowing users to easily test
robust algorithms. They are also available in the R-packages `robustbase` and `rrcov`.

## Hawkings & Bradu & Kass data
```@docs
Robustbase.hbk
```

## Animals data
```@docs
Robustbase.animals
```

## Stack Loss data
```@docs
Robustbase.stackloss
```

## Modified Wood Gravity data
```@docs
Robustbase.wood
```

