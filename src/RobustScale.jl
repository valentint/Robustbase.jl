abstract type RobustScale end

"""
    location(rs::RobustScale)

Univariate location
"""
function location(rs::RobustScale)
    rs.location_ === nothing && throw(ArgumentError("Model not fitted. Call fit!() first."))
    return rs.location_
end

"""
    scale(rs::RobustScale)
    
Univariate scale
"""
function scale(rs::RobustScale)
    rs.scale_ === nothing && throw(ArgumentError("Model not fitted. Call fit!() first."))
    return rs.scale_
end

function fit!(rs::RobustScale, X::Vector{Float64}; ignore_nan::Bool=false)
    if any(isnan, X)
        if rs.can_handle_nan && ignore_nan
            X = filter(!isnan, X)  # Remove NaNs
        else
            throw(ArgumentError("Data contains NaNs, and estimator cannot handle them."))
        end
    end

    _calculate!(rs, X)
    return rs
end

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

"""
    Tau <: RobustScale

    Tau(; c1::Float64=4.5, c2::Float64=3.0, consistency_correction::Bool=true, can_handle_nan::Bool=false)

Robust Tau-Estimate of Scale

Computes the robust τ-estimate of univariate scale, as proposed by Maronna and Zamar (2002); improved by a consistency factor

# Keywords
    `c1::Float64=4.5`, optional: constant for the weight function, defaults to 4.5
    `c2::Float64=3.0`, optional: constant for the rho function, defaults to 3.0
    `consistency_correction::Bool`, optional: boolean indicating if consistency for normality should be applied. Defaults to True.
    `can_handle_nan::Bool`, optional: boolean indicating whether NANs can be handled, defaults to false.
# References
    Robust Estimates of Location and Dispersion for High-Dimensional Datasets,
    Ricarco A Maronna and Ruben H Zamar (2002)
# Examples
```julia
julia> x = hbk[:,1];
julia> tau = Tau();
julia> fit!(tau, x);
julia> location(tau)
1.5554543865072337

julia> scale(tau)
2.012461881814477
```
"""
mutable struct Tau <: RobustScale
    can_handle_nan::Bool
    c1::Float64
    c2::Float64
    consistency_correction::Bool
    location_::Union{Nothing, Float64}
    scale_::Union{Nothing, Float64}

    function Tau(; c1::Float64 = 4.5, c2::Float64 = 3.0, consistency_correction::Bool = true, can_handle_nan::Bool = false)
        new(can_handle_nan, c1, c2, consistency_correction, nothing, nothing)
    end
end

function _calculate!(rs::Tau, X::Vector{Float64})
    n = length(X)
    # sigma0 = mad(X, normalize = false)  # Median Absolute Deviation
    med = median(X)
    sigma0 = median(abs.(X .- med))       # normalize = false 

    weights = weight_function((X .- median(X)) ./ sigma0, rs.c1)
    location = sum(X .* weights) / sum(weights)

    scale = sigma0 * sqrt(sum(rho_function((X .- location) ./ sigma0, rs.c2)) / n)

    if rs.consistency_correction
        # Expectation of rho(X / qnorm(3/4)) for X standard normal
        b = rs.c2 * quantile(Normal(), 3 / 4)
        corr = 2 * ((1 - b^2) * cdf(Normal(), b) - b * pdf(Normal(), b) + b^2) - 1
        scale /= sqrt(corr)
    end

    rs.location_ = location
    rs.scale_ = scale

end

function weight_function(X::Vector{Float64}, c1::Float64)
    return [(abs(x) <= c1) ? (1 - (x / c1)^2)^2 : 0.0 for x in X]
end

function rho_function(X::Vector{Float64}, c2::Float64)
    return [x^2 <= c2^2 ? x^2 : c2^2 for x in X]
end


"""
    Implementation of univariate MCD (Hubert & Debruyne, 2009)

### Keywords
        alpha (float or int, optional): size of the h subset.
          If an integer between n/2 and n is passed, it is interpreted as an absolute value.
          If a float  between 0.5 and 1 is passed, it is interpreted as a proportation
          of n (the training set size).
          If None, it is set to floor(n/2) + 1.
          Defaults to None.
        consistency_correction (boolean, optional):
          whether the estimates should be consistent at the normal model.
          Defaults to True.

### References:
        Hubert, M., & Debruyne, M. (2010). Minimum covariance determinant.
          Wiley interdisciplinary reviews: Computational statistics, 2(1), 36-43.
"""
    mutable struct UnivariateMCD <: RobustScale
    can_handle_nan::Bool
    alpha::Union{Float64, Int, Nothing}
    consistency_correction::Bool
    h_size::Union{Nothing, Int}
    raw_location::Union{Nothing, Float64}
    raw_scale::Union{Nothing, Float64}
    raw_variance::Union{Nothing, Float64}
    location_::Union{Nothing, Float64}
    scale_::Union{Nothing, Float64}
    variance::Union{Nothing, Float64}

    function UnivariateMCD(; alpha::Union{Float64, Int, Nothing} = 0.5, consistency_correction::Bool = true, can_handle_nan::Bool = false)
        new(can_handle_nan, alpha, consistency_correction, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

function _calculate!(rs::UnivariateMCD, X::Vector{Float64})
    rs.h_size = set_h_size(rs.alpha, length(X))
    
    if rs.h_size == length(X)      # return sample location and variance
        rs.raw_location = rs.location_ = mean(X)
        rs.raw_scale = rs.scale_ = std(X)
        rs.raw_variance = rs.variance_ = raw_scale_^2
    end

    rs.raw_variance, rs.raw_location = get_raw_estimates(X, rs.h_size, rs.consistency_correction)
    rs.variance, rs.location_ = reweighting(X, rs.raw_location, rs.raw_variance, rs.consistency_correction)
    rs.raw_scale, rs.scale_ = sqrt(rs.raw_variance), sqrt(rs.variance)
end

"""
    Finds the best subset with minimum variance.
"""
function get_raw_estimates(X::Vector{Float64}, h_size::Int, consistency_correction::Bool=true)
    n = length(X)
    X_sorted = sort(X)

    var_best = Inf
    index_best = 1

    for i in 1:(n - h_size + 1)
        var_new = var(X_sorted[i:(i + h_size - 1)])
        if var_new < var_best
            var_best = var_new
            index_best = i
        end
    end

    raw_var = var_best
    raw_loc = mean(X_sorted[index_best:(index_best + h_size - 1)])

    if consistency_correction
        # [Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]
        factor = (h_size / n) / cdf(Chisq(3), quantile(Chisq(1), h_size / n))
        # println("Factor: ", factor)
        raw_var *= factor
    end

    return raw_var, raw_loc
end

"""
    Refines estimates using chi-squared distribution.
"""
function reweighting(X::Vector{Float64}, raw_location::Float64, raw_variance::Float64, consistency_correction::Bool=true)
    distances = (X .- raw_location) .^ 2 ./ raw_variance
    mask = distances .< quantile(Chisq(1), 0.975)
    loc = mean(X[mask])
    scale = var(X[mask])

    if consistency_correction
        delta = sum(mask) / length(X)
        scale *= delta / cdf(Gamma(3/2), quantile(Chisq(1), delta) / 2)
    end

    return scale, loc
end

"""
   Determines the subset size h_size based on alpha.
"""
function set_h_size(alpha::Union{Float64, Int, Nothing}, n::Int)
    if alpha === nothing
        return div(n, 2) + 1
    elseif alpha isa Int && (n / 2 <= alpha <= n)
        return alpha
    elseif alpha isa Float64 && (0.5 <= alpha <= 1)
        return floor(alpha * n)
    else
        throw(ArgumentError("alpha must be an integer between n/2 and n or a float between 0.5 and 1, but received $alpha"))
    end
end

"""
    weighted_median(X::Vector{Float64}, weights::Vector{Float64})

    Computes a weighted median.

### References
    Time-efficient algorithms for two highly robust estimators
    of scale, Christophe Croux and Peter J. Rousseeuw (1992)
"""
function weighted_median(X::Vector{Float64}, weights::Vector{Float64})
    if length(X) != length(weights)
        throw(ArgumentError("X and weights must have the same length"))
    end

    n = length(X)
    wrest = 0.0
    wtotal = sum(weights)
    Xcand = copy(X)
    weightscand = copy(weights)

    while true
        k = ceil(Int, n / 2)
        if n > 1
            trial = maximum(partialsort(Xcand, 1:k))
        else
            return Xcand[1]
        end

        wleft = sum(weightscand[Xcand .< trial])
        wmid = sum(weightscand[Xcand .== trial])

        if 2 * (wrest + wleft) > wtotal
            mask = Xcand .< trial
        elseif 2 * (wrest + wleft + wmid) > wtotal
            return trial
        else
            mask = Xcand .> trial
            wrest += wleft + wmid
        end

        Xcand = Xcand[mask]
        weightscand = weightscand[mask]
        n = length(Xcand)
    end
end

"""
Calculates the correction factor for the Qn estimator
at small samples [Time-efficient algorithms for two highly robust estimators of scale,
Christophe Croux and Peter J. Rousseeuw (1992)].
"""
function get_small_sample_dn(n::Int)
    DNDICT = Dict(
        2 => 0.399, 3 => 0.994, 4 => 0.512, 5 => 0.844,
        6 => 0.611, 7 => 0.857, 8 => 0.669, 9 => 0.872
    )
    if n <= 9
        return get(DNDICT, n, 1.0)
    elseif isodd(n)
        return n / (n + 1.4)
    end
    return n / (n + 3.8)
end

"""
    Qn <: RobustScale

    Qn(;location_func::Function=median, consistency_correction::Bool=true, finite_correction::Bool=true, can_handle_nan::Bool = false)

Robust Location-Free Scale Estimate More Efficient than MAD

# Keywords
    `location_func::LocationOrScaleEstimator`, optional: as the Qn estimator does not
        estimate location, a location function should be explicitly passed.
    `consistency_correction::Bool`, optional: boolean indicating if consistency for normality should be applied. Defaults to true.
    `finite_correction::Bool`, optional: boolean indicating if finite sample correction should be applied. Defaults to true.

# References
    Christophe Croux and Peter J. Rousseeuw (1992). Time-efficient algorithms for two highly robust estimators of scale.

    Donald B. Johnson and Tetsuo Mizoguchi (1978). Selecting the k^th element in X+Y and X1+...+Xm
# Examples
```julia
julia> x = hbk[:,1];

julia> qn = Qn();
julia> fit!(qn, x);

julia> location(qn)         # the median
1.8

julia> scale(qn)
1.7427832460732984
```
"""
mutable struct Qn <: RobustScale
    can_handle_nan::Bool
    location_func::Function
    consistency_correction::Bool
    finite_correction::Bool
    location_::Float64
    scale_::Float64

    function Qn(;location_func::Function=median, consistency_correction::Bool=true, finite_correction::Bool=true, can_handle_nan::Bool = false)
        new(can_handle_nan, location_func, consistency_correction, finite_correction, NaN, NaN)
    end
end

function _calculate!(q::Qn, X::Vector{Float64})
    n = length(X)
    h = div(n, 2) + 1
    k = div(h * (h - 1), 2)
    y = sort(X)
    left = n .+ 2 .- (1:n)
    right = fill(n, n)
    jhelp = div(n * (n + 1), 2)
    knew = k + jhelp
    nL = jhelp
    nR = n * n
    found = false
    Qn_val = NaN

    while (nR - nL) > n && !found
         weight = right .- left .+ 1
        ##  println("nR, nL, weight: ", nR, ", ", nL, ", ", weight)
        
        jhelp = left .+ div.(weight, 2)
        ##  println("jhelp: ", jhelp)

        ## This is replacement for the Pythons negative indexes:
        ##  i <= 0 ? a[end + i + 1] : a[i]

        ## work = y .- y[n .- jhelp .+ 1]
        indx = n .- jhelp .+ 1
        for ix in 1:length(indx)
            indx[ix] = indx[ix] <= 0 ? length(indx)-indx[ix] : indx[ix]
        end
        work = y .- y[indx]
        trial = weighted_median(work, weight .+ 0.0)
        P = searchsortedfirst.(Ref(-reverse(y)), trial .- y) .- 1
        Q = searchsortedlast.(Ref(-reverse(y)), trial .- y) .+ 1

        if knew <= sum(P)
            right = P
            nR = sum(P)
        elseif knew > (sum(Q) - n)
            left = Q
            nL = sum(Q) - n
        else
            Qn_val = trial
            found = true
        end
    end

    if !found
        work = Float64[]
        for i in 2:n
            if left[i] <= right[i]
                for jj in Int(left[i]):Int(right[i])
                    push!(work, y[i] - y[n - jj + 1])
                end
            end
        end
        k = Int(knew - nL)
        Qn_val = maximum(partialsort(work, 1:k))
    end

    if q.finite_correction
        Qn_val *= get_small_sample_dn(n)
    end
    if q.consistency_correction
        Qn_val *= 2.219144
    end

    q.scale_ = Qn_val
    q.location_ = q.location_func(X)
end

""" 
    MAD_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}
    
Compute the MAD along dimension dims (1=columns, 2=rows)
"""
function MAD_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}
    if dims == 2     # by row
        return [scale(fit!(MAD(), collect(row))) for row in eachrow(X)]
    elseif dims == 1     # by column
        return [scale(fit!(MAD(), collect(col))) for col in eachcol(X)]
    else
        error("dims $dims not supported")
    end
end

""" 
    MAD_scale(X::Vector{Float64})::Float64
    
Compute the MAD along dimension dims (1=columns, 2=rows)
"""
function MAD_scale(X::Vector{Float64})::Float64
    scale(fit!(MAD(), X))
end

""" 
    Qn_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}

Compute the Qn scale along dimension dims (1=columns, 2=rows)
"""
function Qn_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}
    if dims == 2    # by row
        return [scale(fit!(Qn(), collect(row))) for row in eachrow(X)]
    elseif dims == 1     # by column
        return [scale(fit!(Qn(), collect(col))) for col in eachcol(X)]
    else
        error("dims $dims not supported")
    end
end

""" 
    Qn_scale(X::Vector{Float64})::Float64

Compute the Qn scale of collection `X`.
"""
function Qn_scale(X::Vector{Float64})::Float64
    scale(fit!(Qn(), X))
end


""" 
    Tau_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}

Compute the Tau scale along dimension dims (1=columns, 2=rows)
"""
function Tau_scale(X::Matrix{Float64}; dims=1)::Vector{Float64}
    if dims == 2     # by row
        return [scale(fit!(Tau(), collect(row))) for row in eachrow(X)]
    elseif dims == 1     # by column
        return [scale(fit!(Tau(), collect(col))) for col in eachcol(X)]
    else
        error("dims $dims not supported")
    end
end

""" 
    Tau_scale(X::Vector{Float64})::Float64
    
Compute the Tau scale of a collection `X`
"""
function Tau_scale(X::Vector{Float64})::Float64
    scale(fit!(Tau(), X))
end

""" 
    Tau_location(X::Matrix{Float64}; dims=1)::Vector{Float64}

Compute the Tau location along dimension dims (1=columns, 2=rows)
"""
function Tau_location(X::Matrix{Float64}; dims=1)::Vector{Float64}
    if dims == 2     # by row
        return [location(fit!(Tau(), collect(row))) for row in eachrow(X)]
    elseif dims == 1     # by column
        return [location(fit!(Tau(), collect(col))) for col in eachcol(X)]
    else
        error("dims $dims not supported")
    end
end

""" 
    Tau_location(X::Vector{Float64})::Float64

Compute the Tau location of a collection `X`
"""
function Tau_location(X::Vector{Float64})::Float64
    location(fit!(Tau(), X))
end

