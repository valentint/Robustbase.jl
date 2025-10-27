# Robustbase

[![Build Status](https://github.com/valentint/Robustbase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/valentint/Robustbase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov.io](http://codecov.io/github/valentint/Robustbase.jl/coverage.svg?branch=main)](http://codecov.io/github/valentint/Robustbase.jl?branch=main)
[![Doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://valentint.github.io/Robustbase.jl/dev/)

<!-- ![](README-logo.png) -->

The package `Robustbase` provides "Essential" Robust Statistics - tools allowing to analyze data with robust methods: univariate methods, multivariate statistics and regression. We strive to cover the book "Robust Statistics, Theory and Methods (with R)" by 'Maronna, Martin, Yohai and Salibian-Barrera'; Wiley 2019. The package is based on the R packages `robustbase` and `rrcov`.

# Univariate statistics
The classical methods for estimating the parameters of the model may be
affected by outliers. Let's have the following 10 observations. 
The usual way to summarize them is 
to calculate the arithmetic mean (10.49) and the standard deviation (1.68).
If we remove the last observaion (which is visibly quite apart of the rest of the data)
we get for the mean 9.97 and for the standard deviation 0.27. Alternatively, if we compute 
the median, a simple robust location estimate, of all the data and the interquartile range (IQR), 
a robust estimate of the standard deviation, we get 9.98 and 0.17 respectively. 
(Note: Since we expect normally distributed data $x$, for the IQR we use the `iqr()` function from 
`StatsBase` divided by 1.349, i.e. IQRN). Computing the median and IQRN for the first 9 observations gives: 9.98 and 0.13 respectively which is quite close to the estimates using all 10 observations.
Another estimator of the scale is the median absolute deviation given by $MAD(X) = med|x_i−med(X)|$
(multiplied by 1.4833 if we want the MAD to be consistent for the standard deviation at
normally distributed data) - we have 0.21 and 0.18 for all the dat and only 
the 9 regular observations respectively. More examples are given in the table below.

```julia
x = [9.52, 9.68, 10.16, 9.96, 10.08, 9.99, 10.47, 9.91, 9.92, 15.21]
mean(x)
std(x)
```

|               |All 10 observations| Only 9 regular observations |
|---------------|:-----------------:|:---------------------------:|
|$\bar x_n$     |10.49              |9.97                         |
|$median$       |9.98               |9.96                         |
|$\tau-location$|9.96               |9.96                         |
|               |                   |                             |
|$Stdev_n$      |1.68               |0.27                         |
|$IQRN$         |0.17               |0.13                         |
|$MAD$          |0.21               |0.18                         |
|$\tau-scale$   |0.28               |0.22                         |
|$Q_n-scale$    |0.37               |0.31                         |
 
The univariate module contains several robust univariate location and scale estimators. 
They all implement the abstract base class `RobustScale`, which has
the properties `location` and `scale` and can be fitted using the `fit!` method. Each 
class is expected to implement a `_calculate` method where the attributes `scale_` and
`location_` are set.

Let us have a univariate dataset $X = \{x_1, \ldots , x_n\}$ of size $n$. As we saw above, 
a simple location estimator is the median of the dataset $med(X)$, and the scale
can be estimated by the median absolute deviation. 
A simple class that uses the median for location and
the MAD for scale would look like this:
```
using Statistics
mutable struct MAD <: RobustScale
    can_handle_nan::Bool
    location_::Union{Nothing, Float64}
    scale_::Union{Nothing, Float64}

    function MAD(; can_handle_nan::Bool = false)
        new(can_handle_nan, nothing, nothing)
    end
end

function _calculate!(rs::MAD, X::Vector{Float64})
    med = median(X)
    madtemp = median(abs.(X .- med)) * 1.4826  # Scale factor for normal consistency
    rs.location_ = med
    rs.scale_ = madtemp
end
```
This estiamtor can be used as follows:
```
using Random
Random.seed!(1234)
x = randn(10)
mad = MAD();
_calculate!(mad, x);
location(mad)
##  0.3689500229232851
scale(mad)
## 1.164914811753848
```
All functions have matrix versions which can be called on a matrix or a data frame
specifying the required dimension on which to do the calculations.
```
using Random
Random.seed!(1234)
x = randn(10,3)

Tau_location(x)         # columnwise Tau-location
Tau_location(x, dims=2) # rowwise Tau-location
Tau_scale(x)            # columnwise Tau-scale
Tau_scale(x, dims=2)    # rowwise Tau-scale
Qn_scale(x)             # columnwise Qn-scale
Qn_scale(x, dims=2)     # rowwise Qn-scale
```

## Qn robust estimator
One of the scale estiamtors included in the package is 
the `Qn` scale estimator of Rousseeuw and Croux (1993). 
It is defined as the first Quartile of the distances 
between the points. This can be written as follows, 
```math
Qn = 2.219\{|x_i-x_j|:i<j\}_{(k)}
```
with $k=\left(\frac{h}{2}\right)$ for $h=\lfloor\frac{n}{2}\rfloor + 1$. 
It has much better statistical efficiency than the MAD, and is computed by the fast algorithm
of Croux and Rousseeuw (1992).

## Tau location and scale estimator
Another scale estimator included in the package is the $\tau$-estimator from Maronna and Yohai (1992).
It is a special case of the one-step M-estimators given by
```math
\tau_{location} = \frac{\sum_i{w_ix_i}}{\sum_i{w_i}} \text{~~~and~~~} 
\tau_{scale} = \sqrt{\frac{MAD^2(X)}{n} \sum_i{\rho_{c2}} \left(\frac{x_i-\tau_{location}}{MAD(X)} \right)} 
```
where the weights $w_i$ are defined as
```math
w_i = W_{c1} \left( \frac{x_i-med(X)}{MAD(X)} \right) \text{~~~with~~~} W_c(u)=\left( 1-\left(\frac{u}{c} \right)^2\right)^2 I(|u| \le c)
```
and $\rho_c(u) = min(c^2, u^2)$. The default values are $c_1 = 4.5$ and $c_2 = 3$, but different values
can be provided to the constructor of the `Tau` object.
# Covariance
Similarly as in the univariate case outliers can influence the 
estimators of multivariate data, these are in first line the 
multivariate location and covariance esimators. Apart from 
being useful for outlier detection through computing the Mahalanobis distances,
they are cornerstones of many other multivariate methods 
like principal component analysis and discriminant analysis.
The most popular robust estimator of multivariate location 
and covariance is the Minimum Covariance Determinant (MCD) 
estimator of Rousseeuw (1084) which is widely used 
after the fast algorithm of Rousseeuw and van Driessen (1999) became 
available. A faster version of the MCD is the deterministic MCD of 
Hubert, Rousseeuw and Verdonck (2012) which instead of doing 
time consuming resampling starts from six rough robust 
estimators which are easy to compute. 

The covariance module implements curently three frequently 
used scatter estimators: The Fast MCD, the deterministic MCD and the OGK.
They all use the new abstract base class `RobustCovariance`
which extends the `CovarianceEstimator` class from `StatsBase`.
The RobustCovariance class includes the distance-distance plot. 
It shows the robust distances versus the classical Mahalanobis 
distances, and is equipped with thresholds for outlier detection.
The plot is drawn by the `dd_plot()` function
function, after obtaining a robust covariance estimator by the `fit!()` 
method.

### Example

```{julia}
mcd = CovMcd();
fit!(mcd, hbk[:, 1:3]);
display(mcd)
dd_plot(mcd)
```
```
-> Method:  Fast MCD Estimator: (alpha=0.5 ==> h=39)

Robust estimate of location:
[1.55833, 1.80333, 1.66]

Robust estimate of covariance:
3×3 Matrix{Float64}:
 1.21312    0.0239154  0.165793
 0.0239154  1.22836    0.195735
 0.165793   0.195735   1.12535
```
![](dd_plot.png)<!-- -->


# Regression

Not yet implemented

# Principal Component Analysis

Not yet implemented
