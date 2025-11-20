    ##  using LinearAlgebra, Random, Statistics, Distributions, Logging
##  using DataFrames
##  using Plots

function fast_cov(X::Matrix{Float64})
    n = size(X, 1)
    mu = mean(X, dims=1)
    centered = X .- mu
    return (centered' * centered) / (n - 1)
end

function mahalanobis_distance(data::Matrix{Float64}, location::Vector{Float64}, covariance::Matrix{Float64})

    cov_inv = inv(covariance)
    centered_data = data .- location'

    return sqrt.(sum((centered_data * cov_inv) .* centered_data, dims=2))[:, 1]
end

function get_logger(name::String; level::Logging.LogLevel = Logging.Info)
    logger = ConsoleLogger(stderr, level)
    global_logger(logger)
    return logger
end

## abstract type CovarianceEstimator end - from StatsBase

function fit!(model::CovarianceEstimator, X::Union{Matrix{Float64}, Vector{Float64}, DataFrame})
    if X isa Vector{Float64}
        X = reshape(X, (:,1))
    elseif X isa DataFrame
        X = Matrix(X)
    end

    ## Handle here missing data ...

    model.X = X
    model.nobs = size(X, 1)
    
    calculate_covariance!(model, X)
    model.mahalanobis_distances_ = mahalanobis_distance(X, model.location_, model.covariance_)

return model
end

function _fitted_covariance(model)::Bool
    if !hasproperty(model, :covariance_)
        error("Model has no 'covariance' property!")
    end
    if isnothing(model.covariance_) || isempty(model.covariance_)
        error("Model is not fitted yet!")
    end
    return true
end

"""
    location(model::CovarianceEstimator)::Vector{Float64}

Location vector
"""
function location(model::CovarianceEstimator)::Vector{Float64}
    if isnothing(model.location_) || length(model.location_) == 0
        error("Model is not fitted yet!")
    end
    return model.location_
end

"""
    covariance(model::CovarianceEstimator)::Matrix{Float64}

Covariance matrix
"""
function covariance(model::CovarianceEstimator)::Matrix{Float64}
    _fitted_covariance(model)
    return model.covariance_
end

"""
    correlation(model::CovarianceEstimator)::Matrix{Float64}

Correlation matrix
"""
function correlation(model::CovarianceEstimator)::Matrix{Float64}
    _fitted_covariance(model)
    return cov2cor(covariance(model))
end

"""
    distance(model::CovarianceEstimator)::Vector{Float64}

Mahalanobis distance
"""
function distance(model::CovarianceEstimator)::Vector{Float64}
    _fitted_covariance(model)
    return model.mahalanobis_distances_
end

"""
    CovClassic <: CovarianceEstimator <: Any

    CovClassic(;assume_centered=false)

Classical Location and Scatter Estimation

Compute the classical estimetas of the location vector and covariance matrix of a data matrix.

# Examples
```julia
julia> cc=CovClassic();
julia> fit!(cc, hbk[:,1:3])
-> Method:  Classical Estimator. 

Estimate of location:
[3.20667, 5.59733, 7.23067]

Estimate of covariance:
3×3 Matrix{Float64}:
 13.3417  28.4692   41.244
 28.4692  67.883    94.6656
 41.244   94.6656  137.835
```
"""
mutable struct CovClassic <: CovarianceEstimator
    nobs::Int
    location_::Union{Vector{Float64}, Nothing}
    covariance_::Union{Matrix{Float64}, Nothing}
    precision_::Union{Matrix{Float64}, Nothing}
    assume_centered::Bool
    mahalanobis_distances_::Vector{Float64}
    X::Union{Matrix{Float64}, Nothing}      # The input data matrix without NAs

    function CovClassic(; assume_centered::Bool = false)
        new(0, nothing, nothing, nothing, assume_centered, [], nothing)
    end
end

function calculate_covariance!(model::CovClassic, X::Matrix{Float64})

    if model.assume_centered
        model.location_ = zeros(size(X, 2))
    else
        model.location_ = mean(X, dims=1)[:]
    end

    model.covariance_ = cov(X, dims=1)
    model.precision_ = inv(model.covariance_)

    return model
end

function Base.show(io::IO, mime::MIME"text/plain", obj::CovClassic)
    println(io, "-> Method:  Classical Estimator. ")
    if isnothing(obj.covariance_)
        println()
        println("Model is not fitted yet!")
    else
        println()
        println(io, "Estimate of location:")
        println(IOContext(stdout, :compact=>true), location(obj))
        println()
        println(io, "Estimate of covariance:")
        Base.show(stdout, mime, covariance(obj))
    end
end

abstract type RobustCovariance <: CovarianceEstimator end

function fit!(model::RobustCovariance, X::Union{Matrix{Float64}, Vector{Float64}, DataFrame})
    if X isa Vector{Float64}
        X = reshape(X, (:,1))
    elseif X isa DataFrame
        X = Matrix(X)
    end

    ## Handle here missing data ...

    model.X = X
    model.nobs = size(X, 1)

    if model.assume_centered
        model.default_location_ = zeros(size(X, 2))
    else
        model.default_location_ = mean(X, dims=1)[:]
    end
    model.default_covariance_ = cov(X, dims=1)

    calculate_covariance!(model, X)

    model.robust_distances_ = mahalanobis_distance(X, model.location_, model.covariance_)
    model.mahalanobis_distances_ = mahalanobis_distance(X, model.default_location_, model.default_covariance_)

    return model
end

"""
    distance(model::RobustCovariance)::Vector{Float64}

Robust Mahalanobis distance
"""
function distance(model::RobustCovariance)::Vector{Float64}
    _fitted_covariance(model)
    return model.robust_distances_
end

struct HSubset
    indices::Vector{Int}
    location::Vector{Float64}
    scale::Matrix{Float64}
    determinant::Float64
    n_c_steps::Int
end

"""
    _get_subset(indices::Vector{Int}, X::Matrix{Float64}; n_c_steps::Int=0)::HSubset

Construct an HSubset from a set of data indices and calculate location, scale and determinant.

# Arguments:
    - indices: data indices
    - X: complete dataset
    - n_c_steps: will be passed directly to the HSubset

# Returns:
    A new Hsubset
"""
function _get_subset(indices::Vector{Int}, X::Matrix{Float64}; n_c_steps::Int=0)::HSubset
    
    logdet_Lrg = 10
    n = size(X, 1)
    p = size(X, 2)
    mu = mean(X[indices, :], dims=1)[:]
    ##  cov_matrix = cov(X[indices, :], dims=1)
    cov_matrix = fast_cov(X[indices, :])
    det_value = logabsdet(cov_matrix)[1]

    ## while isapprox(det_value, 0, atol=1e-9)
    while -det_value/p > logdet_Lrg 
        if length(indices) == n
            error("All subsamples are singular! Data set is not in general position.")
        end
        new_index = rand(setdiff(1:n, indices))
        ## replace by push!()
        ##  indices = vcat(indices, new_index)      
        push!(indices, new_index)
        mu = mean(X[indices, :], dims=1)[:]
        ##  cov_matrix = cov(X[indices, :], dims=1)
        cov_matrix = fast_cov(X[indices, :])
        det_value = logabsdet(cov_matrix)[1]
    end

    return HSubset(indices, mu, cov_matrix, det_value, n_c_steps)
end

"""
    h_alpha_n(alpha::Union{Float64, Int}, n::Int, p::Int)::Int  

Compute h(alpha) which is the size of the subsamples to be used for MCD and LTS. 
Given alpha, n and p, h is an integer, h approx alpha n, where the 
exact formula also depends on p.

For alpha = 1/2, h == floor(n+p+1)/2.
For the general case, it's simply n2 = div(n+p+1, 2); floor(2 * n2 - n + 2 * (n-n2) * alpha).
"""
function h_alpha_n(alpha::Union{Float64, Int}, n::Int, p::Int)::Int 
    ## Compute h(alpha) := size of subsample, given alpha, (n,p)
    ## Same function for covMcd() and ltsReg()
    n2 = div(n+p+1, 2)
    return floor(2 * n2 - n + 2 * (n - n2) * alpha)
end

"""
    _get_h(alpha::Union{Nothing, Float64, Int}, X::Matrix{Float64})::Int

    Determines the subset size h based on parameter alpha and the 
    shape of the data (n,p).
"""
function _get_h(alpha::Union{Nothing, Float64, Int}, X::Matrix{Float64})::Int
    if isnothing(alpha)
        ret = h_alpha_n(0.5, size(X, 1), size(X, 2))
        # model.alpha = round(ret/size(X, 1), sigdigits=2)
        return ret
    elseif alpha isa Int && (div(size(X, 1), 2) <= alpha <= size(X, 1))
        return alpha
    elseif (alpha isa Float64 && (0.5 <= alpha <= 1)) || alpha == 1
        return h_alpha_n(alpha, size(X, 1), size(X, 2))
    else
        error("Invalid alpha value: $(alpha). Must be between n/2 and n (integer) or between 0.5 and 1 (float)!")
    end
end

"""
    Perform a single C-step on the subset of the data

    Args:
        h: size of subset
        subset: initial H-subset
        X: data

    Returns:
        indices for new h subset
"""
function _perform_c_step(h::Int, subset::HSubset, X::Matrix{Float64})::HSubset
    mahalanobis = mahalanobis_distance(X, subset.location, subset.scale)
    idx = collect(partialsortperm(mahalanobis, 1:h))
    return _get_subset(idx, X; n_c_steps=subset.n_c_steps + 1)
end

"""
    Compute a consistency correction factor, depending on the number of variables p and 
    the size of the half sample alpha=h/n, to make the MCD estimate consistent at the 
    normal model. The rescaling factor is returned in the length-2 vector raw.cnp2.
"""
function MCDcons(p::Int, alpha::Float64)
    ## qalpha <- qchisq(alpha, p)
    ## caI <- pgamma(qalpha/2, p/2 + 1) / alpha
    ## 1/caI

    ## quantile of chi-square distribution with p degrees of freedom
    qalpha = quantile(Chisq(p), alpha)
    
    ##  pgamma(x, shape) in R → CDF of a Gamma distribution with shape and scale = 1. 
    ##  Julia’s Distributions.Gamma has the same convention, so we can use cdf(Gamma(shape, 1), x).
    caI = cdf(Gamma(p/2 + 1, 1), qalpha/2) / alpha
    return 1/caI
end

"""
    CovMcd <: RobustCovariance <: CovarianceEstimator 

    CovMcd(;assume_centered=false, alpha=nothing, n_initial_subsets=500, n_initial_c_steps=2,
        n_best_subsets=10, n_partitions=nothing, tolerance=1e-8, reweighting=true, verbosity=Logging.Warn)

Robust Location and Scatter Estimation via MCD

Compute the Minimum Covariance Determinant (MCD) estimator, a robust multivariate location and scale 
estimate with a high breakdown point, via the 'Fast MCD' algorithm proposed in Rousseeuw and Van Driessen (1999).

# Keywords:

    `alpha::Float64 | Int | Nothing`, optional:
        size of the h subset.
        If an integer between n/2 and n is passed, it is interpreted as an absolute value.
        If a float between 0.5 and 1 is passed, it is interpreted as a proportation
        of n (the training set size).
        If None, it is set to (n+p+1) / 2.
        Defaults to Nothing.
    `n_initial_subsets::Int`, optional: number of initial random subsets of size p+1
    `n_initial_c_steps::Int`, optional: number of initial c steps to perform on all initial subsets
    `n_best_subsets::Int`, optional: number of best subsets to keep and perform c steps on until convergence
    `n_partitions::Int` optional: Number of partitions to split the data into.
        This can speed up the algorithm for large datasets (n > 600 suggested in paper)
        If None, 5 partitions are used if n > 600, otherwise 1 partition is used.
    `tolerance::Float64`, optional: Minimum difference in determinant between two iterations to stop the C-step
    `reweighting:Bool`, optional: Whether to apply reweighting to the raw covariance estimate

# References:
    Rousseeuw and Van Driessen, A Fast Algorithm for the Minimum Covariance Determinant
    Estimator, 1999, American Statistical Association and
    the American Society for Quality, TECHNOMETRICS

# Examples
```julia
julia> mcd=CovMcd();
julia> fit!(mcd, hbk[:,1:3])
-> Method:  Fast MCD Estimator: (alpha=nothing ==> h=39)

Robust estimate of location:
[1.55833, 1.80333, 1.66]

Robust estimate of covariance:
3×3 Matrix{Float64}:
 1.21312    0.0239154  0.165793
 0.0239154  1.22836    0.195735
 0.165793   0.195735   1.12535

juilia> dd_plot(mcd)
```
"""
mutable struct CovMcd <: RobustCovariance
    alpha::Union{Nothing, Float64, Int}
    n_initial_subsets::Int
    n_initial_c_steps::Int
    n_best_subsets::Int
    n_partitions::Union{Nothing, Int}
    tolerance::Float64
    reweighting::Bool
    verbosity::Base.CoreLogging.LogLevel
    assume_centered::Bool
    nmini::Int
    kmini::Int
    nobs::Int
    quan::Int
    best::Vector{Int}
    crit::Float64
    location_::Vector{Float64}
    covariance_::Matrix{Float64}
    default_location_::Vector{Float64}
    default_covariance_::Matrix{Float64}
    raw_location::Vector{Float64}
    raw_covariance::Matrix{Float64}
    robust_distances_::Vector{Float64}
    mahalanobis_distances_::Vector{Float64}
    raw_distances::Vector{Float64}
    best_subset::HSubset
    raw_cnp2::Vector{Float64}
    cnp2::Vector{Float64}
    X::Union{Matrix{Float64}, Nothing}      # The input data matrix without NAs

    function CovMcd(;assume_centered=false, alpha=nothing, n_initial_subsets=500, n_initial_c_steps=2,
        n_best_subsets=10, n_partitions=nothing, tolerance=1e-8,
        reweighting=true, verbosity=Logging.Warn)

        if alpha isa Float64 && (0.5 > alpha || alpha >= 1)
            error("Invalid alpha value: $(alpha). Must be between 0.5 and 1 (float)!")
        end

        new(alpha, n_initial_subsets, n_initial_c_steps, n_best_subsets, 
            n_partitions, tolerance, reweighting, verbosity, assume_centered, 300, 5, 
            0, 0, [], 0, [], [;;], [], [;;], [], [;;], [], [], [], 
            HSubset([], [], [;;], 0.0, 0), [], [], nothing)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", obj::CovMcd)
    if isnothing(obj.covariance_) || size(obj.covariance_, 1) == 0
        println(stdout, "-> Method:  Fast MCD Estimator")
        println()
        println("Model is not fitted yet!")
    else
        alpha = if(obj.alpha == nothing) 0.5 else obj.alpha end
        println(stdout, "-> Method:  Fast MCD Estimator: (alpha=", alpha, " ==> h=", obj.quan, ")")
        println()
        println(io, "Robust estimate of location:")
        println(IOContext(stdout, :compact=>true), location(obj))
        println()
        println(io, "Robust estimate of covariance:")
        Base.show(stdout, mime, covariance(obj))
    end
end

function calculate_covariance!(model::CovMcd, X::Matrix{Float64})
    
    n, p = size(X)
    alpha = model.alpha
    if alpha == 1 || alpha == n
        @warn "Default covariance is returned as alpha is $(alpha)."
        model.location_ = mean(X, dims=1)[:]
        model.covariance_ = cov(X, dims=1)
        model.quan = n
        return model
    end

    h = model.quan = _get_h(model.alpha, X)
    ## @info "Called _get_h() from calculate_covariance(): h=", h, "alpha=", model.alpha
    partitions = _partition_data(model, X)
    ##  @info "Partitioned data into $(length(partitions)) partitions"
    n_initial_subsets = div(model.n_initial_subsets, length(partitions))

    best_subsets = HSubset[]
    
    # Step 1: Perform initial C-steps on all initial subsets
    #=
    for data in partitions
        subsets = _get_initial_subsets(model, data, n_initial_subsets)
        # println("Initial subset: ", subsets[1].indices)
        subsets = [_perform_multiple_c_steps(model, subset, data, model.n_initial_c_steps) for subset in subsets]
        best_subsets = vcat(best_subsets, sort(subsets, by=x -> x.determinant)[1:model.n_best_subsets])
    end
    =#

    # Step 1: Perform initial C-steps on all initial subsets
    best_subsets = Vector{HSubset}(undef, model.n_best_subsets * length(partitions))
    i = 1
    for data in partitions
        subsets = _get_initial_subsets(model, data, n_initial_subsets)
        subsets = [_perform_multiple_c_steps(model, subset, data, model.n_initial_c_steps) for subset in subsets]
        top = sort(subsets, by=x -> x.determinant)[1:model.n_best_subsets]
        for s in top
            best_subsets[i] = s
            i += 1
        end
    end
    resize!(best_subsets, i - 1)  # truncate to actual size

    ## @info "Step 1: Selecting $(model.n_best_subsets) best subsets from $(length(best_subsets))"
    
    # Step 2: Perform additional C-steps on the best subsets
    best_subsets = [_perform_multiple_c_steps(model, subset, X, model.n_initial_c_steps) for subset in best_subsets]
    best_subsets = sort(best_subsets, by=x -> x.determinant)[1:model.n_best_subsets]
    best_subset = first(best_subsets)  # Initial best subset

    # Step 3: Perform C-steps until convergence
    for subset in best_subsets
        while true
            new_subset = _perform_c_step(model.quan, subset, X)
            if new_subset.indices == subset.indices || (new_subset.determinant - subset.determinant) < model.tolerance
                break
            end
            subset = new_subset
        end
        if subset.determinant < best_subset.determinant
            best_subset = subset
        end
    end

    model.best_subset = best_subset

    # Post-processing
    model.raw_location = best_subset.location
    model.raw_cnp2 = [MCDcons(p, h/n), 1]
    model.raw_covariance = best_subset.scale * model.raw_cnp2[1] * model.raw_cnp2[2]
    ##  @info "Correction factor - raw estimate", model.raw_cnp2[1], model.raw_cnp2[2]
    model.raw_distances = mahalanobis_distance(X, model.raw_location, model.raw_covariance)
    model.best = sort(model.best_subset.indices)
    model.crit = model.best_subset.determinant

    if model.reweighting
        @debug "Applying reweighting to the raw covariance estimate"
        mask = model.raw_distances .< sqrt(quantile(Chisq(size(X, 2)), 0.975))
        model.location_ = mean(X[mask, :], dims=1)[:]
        model.covariance_ = cov(X[mask, :], dims=1)

        @debug "Applying consistency correction after reweighting."
        model.cnp2 = [MCDcons(p, 0.975), 1]
        model.covariance_ *= model.cnp2[1] * model.cnp2[2]
        ##  @info "Correction factor - reweighted estimate", model.cnp2[1], model.cnp2[2]
    else
        model.location_ = model.raw_location
        model.covariance_ = model.raw_covariance
        model.cnp2 = [1, 1]
    end

    return model
end

"""
Splits the data into partitions if necessary (n > 600).
"""
function _partition_data(model::CovMcd, X::Matrix{Float64})::Vector{Matrix{Float64}}
    
    n = size(X, 1)
    if n < 2 * model.nmini      # i.e. less than 600
        return [X]
    end
    nmini = model.nmini
    kmini = model.kmini
    ngroup = fld(n, nmini)

    if ngroup < kmini
        mm = fld(n, ngroup)
        r = n - ngroup * mm
        jj = ngroup - r
        mini = vcat(fill(mm, jj), fill(mm+1, ngroup-jj))
        minigr = ngroup * mm + r
    else
        ngroup = kmini
        mini = fill(nmini, kmini)
        minigr = kmini * nmini
    end

    id = randperm(n)[1:minigr]

    # Step 1: Create zero-copy views
    parts_views = Vector{SubArray}(undef, ngroup)
    kk = 1
    for i in 1:ngroup
        inds = id[kk:(kk + mini[i] - 1)]
        parts_views[i] = @view X[inds, :]
        kk += mini[i]
    end

    # Step 2: Materialize copies → now type is Vector{Matrix}
    return [Matrix(v) for v in parts_views]
end

#=
This is the old version, as in Python,  which does not work!
It takes always 5 groups (if n >= 600) and almost always remain several observations
(the rest of the divisin of n by 5) in a 6-th partition.
 
function _partition_data(model::CovMcd, X::Matrix{Float64})::Vector{Matrix{Float64}}
    n_partitions = isnothing(model.n_partitions) ? (size(X, 1) > 600 ? 5 : 1) : model.n_partitions
    return [X[i:min(i+div(size(X,1), n_partitions)-1, size(X,1)), :] for i in 1:div(size(X,1), n_partitions):size(X,1)]
end
=#


"""
Repeatedly applies the C-step (n_iterations times).
"""
function _perform_multiple_c_steps(model::CovMcd, subset::HSubset, X::Matrix{Float64}, n_iterations::Int)::HSubset
    h = _get_h(model.alpha, X)
    for _ in 1:n_iterations
        subset = _perform_c_step(h, subset, X)
    end
    return subset
end

"""
Generates initial candidate subsets (n_subsets) and return a vector of HSubset[n_subsets]
    with indices of the subset, location, covariance and determinant of the covariance.
"""
function _get_initial_subsets(model::CovMcd, X::Matrix{Float64}, n_subsets::Int)::Vector{HSubset}
    return [_get_subset(sample(1:size(X, 1), size(X, 2) + 1, replace=false), X) for _ in 1:n_subsets]
end

#################################################################################################################

function sqrtm_symmetric(M::AbstractMatrix)
    # Check symmetric
    ##  issymmetric(M) || error("Matrix must be symmetric")

    # Eigendecomposition: M = Q * Diag(λ) * Q'
    eigen_decomp = eigen(M)
    Q = eigen_decomp.vectors
    λ = eigen_decomp.values

    # Replace negative/zero eigenvalues with zero to ensure sqrt is real and safe
    λ_sqrt = sqrt.(clamp.(λ, 0, Inf))

    # Matrix square root: Q * Diag(sqrt(λ)) * Q'
    return Q * Diagonal(λ_sqrt) * Q'
end

function rank_gaussian_transform(R::AbstractMatrix)
    n, p = size(R)
    transformed = Array{Float64}(undef, n, p)
    for j in 1:p
        transformed[:, j] .= quantile.(Normal(), ((R[:, j] .- 1/3) ./ (n + 1/3)))
    end
    return transformed
end

"""
    DetMcd(; assume_centered=false,  alpha=nothing, n_maxcsteps=200, tolerance=1e-8,
        reweighting=true, verbosity=Logging.Warn)

Deterministic MCD estimator (DetMCD) based on the algorithm proposed in
Hubert, Rousseeuw and Verdonck (2012)

# Keywords:
    `alpha::Float64 | Int | Nothing`, optional: size of the h subset.
        If an integer between n/2 and n is passed, it is interpreted as an absolute value.
        If a float between 0.5 and 1 is passed, it is interpreted as a proportation
        of n (the training set size).
        If None, it is set to (n+p+1) / 2.
        Defaults to None.
    `n_maxcsteps::Int=200`, optional: Maximum number of C-step iterations
    `tolerance::Float64`, optional: Minimum difference in determinant between two iterations to stop the C-step
    `reweighting::Bool`, optional: Whether to apply reweighting to the raw covariance estimate

# References:
    Hubert, Rousseeuw and Verdonck, A deterministic algorithm for robust location
    and scatter, 2012, Journal of Computational and Graphical Statistics

# Examples
```julia
julia> mcd=DetMcd();
julia> fit!(mcd, hbk[:,1:3])
-> Method:  Deterministic MCD: (alpha=nothing ==> h=39)

Robust estimate of location:
[1.5377, 1.78033, 1.68689]

Robust estimate of covariance:
3×3 Matrix{Float64}:
 1.2209     0.0547372  0.126544
 0.0547372  1.2427     0.151783
 0.126544   0.151783   1.15414

julia> dd_plot(mcd);
```
"""
mutable struct DetMcd <: RobustCovariance
    alpha::Union{Nothing, Float64, Int}
    n_maxcsteps::Int
    tolerance::Float64
    reweighting::Bool
    verbosity::Base.CoreLogging.LogLevel
    assume_centered::Bool
    nobs::Int
    quan::Int
    best::Vector{Int}
    iBest::Vector{Int}
    crit::Float64
    location_::Union{Nothing, Vector{Float64}}
    covariance_::Union{Nothing, Matrix{Float64}}
    default_location_::Vector{Float64}
    default_covariance_::Matrix{Float64}
    raw_location::Vector{Float64}
    raw_covariance::Matrix{Float64}
    robust_distances_::Vector{Float64}
    mahalanobis_distances_::Vector{Float64}
    raw_distances::Vector{Float64}
    best_subset::Union{Nothing, HSubset}
    raw_cnp2::Vector{Float64}
    cnp2::Vector{Float64}
    mask::Union{Nothing, Vector{Bool}}
    initHsets::Matrix{Int}
    X::Union{Matrix{Float64}, Nothing}      # The input data matrix without NAs

    function DetMcd(; assume_centered=false,  alpha=nothing, n_maxcsteps=200, tolerance=1e-8,
        reweighting=true, verbosity=Logging.Warn)
        new(alpha, n_maxcsteps, tolerance, reweighting, verbosity, assume_centered,
        0, 0, [], [], 0, [], [;;], [], [;;], [], [;;], [], [], [], HSubset([], [], [;;], 0.0, 0), [], [], [], [;;], nothing)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", obj::DetMcd)
    if isnothing(obj.covariance_) || size(obj.covariance_, 1) == 0
        println(stdout, "-> Method:  Deterministic MCD")
        println()
        println("Model is not fitted yet!")
    else
        alpha = if(obj.alpha == nothing) 0.5 else obj.alpha end
        println(stdout, "-> Method:  Deterministic MCD: (alpha=", alpha, " ==> h=", obj.quan, ")")
        println()
        println(io, "Robust estimate of location:")
        println(IOContext(stdout, :compact=>true), location(obj))
        println()
        println(io, "Robust estimate of covariance:")
        Base.show(stdout, mime, covariance(obj))
    end
end

function calculate_covariance!(model::DetMcd, X::Matrix{Float64})
    n, p = size(X)
    alpha = model.alpha

    if alpha == 1 || alpha == n
        @warn "Default covariance is returned as alpha is $alpha."
        model.location_ = mean(X, dims=1)[:]
        model.covariance_ =  cov(X, dims=1)
        model.quan = n
        return model
    end

    h = model.quan = _get_h(model.alpha, X)

    # Step 0: standardize X
    Z_center = median(X, dims=1)
    Z_scale = n < 1000 ? Qn_scale(X) : Tau_scale(X)
    Z = (X .- Z_center) ./ Z_scale' 

    ##  @info "Standartized Matrix"
    ##  display("text/plain", Z)

    # Steps 1-3:
    best_subsets = get_initial_best_subsets(model, Z, X, n, p)

    # Step 4: C-steps
    model.initHsets = zeros(h, 6)
    best_subset = best_subsets[1]
    iBest = []
    i = 0
    for subset in best_subsets
        i += 1
        jj = 0
        for j=1:model.n_maxcsteps
            jj += 1
            new_subset = _perform_c_step(model.quan, subset, X)
            if new_subset.indices == subset.indices 
            ## if new_subset.indices == subset.indices || (new_subset.determinant - subset.determinant) < model.tolerance
                break
            end
            subset = new_subset
        end

        ##  @info "C-steps: ", i, jj, subset.determinant, subset.n_c_steps 

        model.initHsets[:,i] = sort(subset.indices)
        if subset.determinant < best_subset.determinant
            best_subset = subset
            iBest = [i]
        elseif subset.determinant == best_subset.determinant
            append!(iBest, i)
        end
    end

    model.iBest = iBest
    model.best_subset = best_subset
    model.location_ = best_subset.location
    model.covariance_ = best_subset.scale

    # Post-processing

    model.raw_location = best_subset.location
    model.raw_covariance = best_subset.scale

    model.raw_cnp2 = [MCDcons(p, h/n), 1]
    model.raw_covariance = best_subset.scale * model.raw_cnp2[1] * model.raw_cnp2[2]
    ##  @info "Correction factor - raw estimate", model.raw_cnp2[1], model.raw_cnp2[2]
    
    model.raw_distances = mahalanobis_distance(X, model.raw_location, model.raw_covariance)
    model.best = sort(model.best_subset.indices)
    model.crit = model.best_subset.determinant

    if model.reweighting
        @debug "Applying reweighting to the raw covariance estimate"
        mask = model.raw_distances .< sqrt(quantile(Chisq(size(X, 2)), 0.975))
        model.location_ = mean(X[mask, :], dims=1)[:]
        model.covariance_ = cov(X[mask, :], dims=1)

        @debug "Applying consistency correction after reweighting."
        model.cnp2 = [MCDcons(p, 0.975), 1]
        model.covariance_ *= model.cnp2[1] * model.cnp2[2]
        ##  @info "Correction factor - reweighted estimate", model.cnp2[1], model.cnp2[2]
    else
        model.location_ = model.raw_location
        model.covariance_ = model.raw_covariance
        model.cnp2 = [1, 1]
    end

    return model
end

function get_initial_best_subsets(model::DetMcd, Z::Matrix{Float64}, X::Matrix{Float64}, n::Int, p::Int)
    # Step 1: construct 6 preliminary estimates Sₖ

    ## 1. Hyperbolic tangent of standardized data
    Y = tanh.(Z)
    S1 = cor(Y, dims=1)

    ## 2. Spearmann correlation matrix
    R = hcat(tiedrank.(eachcol(Z))...)
    S2 = cor(R, dims=1)

    ## 3. Tukey normal scores
    ##  R_scaled = [quantile.(Normal(), ((R[:, j] .- 1/3) ./ (n + 1/3))) for j in 1:p] |> hcat
    R_scaled = rank_gaussian_transform(R)
    S3 = cor(R_scaled, dims=1)

    ## 4. Spatial sign covariance matrix
    znorm = sqrt.(sum(Z .^ 2, dims=2))[:]
    ii = znorm .> eps()     # Machine epsilon (Float64)
    Zw = copy(Z)            # This copy is essential - otherwise only reference is taken
    Zw[ii,:] = Zw[ii,:] ./ znorm[ii]
    S4 = (Zw' * Zw)
    
    ## 5. BACON
    idx = sortperm(znorm)[1:ceil(Int, n / 2)]
    S5 = cov(Z[idx, :], dims=1)

    ## 6. Raw OGK estimate for scatter
    if size(Z, 2) == 1      # X is one-dimensional, OGK cannot be used!
        S6 = S5
    else
        ogk = CovOgk(location_estimator=median, scale_estimator=Tau_scale, reweighting=false)
        S6 = calculate_covariance!(ogk, Z)
    end

    estimates_S = [S1, S2, S3, S4, S5, S6]

    #=
    @info "S1:==============================="
    display("text/plain", S1)
    @info "S2:==============================="
    display("text/plain", S2)
    @info "S3:==============================="
    display("text/plain", S3)
    @info "S4:==============================="
    display("text/plain", S4)
    @info "S5:==============================="
    display("text/plain", S5)
    @info "S6:==============================="
    display("text/plain", S6)
    =#

    # Step 2: construct 6 initial location and scatter estimates
    estimates_mu = Vector{Vector{Float64}}()
    estimates_sigma = Vector{Matrix{Float64}}()

    ix = 1
    for S in estimates_S
        E = eigen(S)
        E = reverse(E.vectors; dims=2)  # Sort eigenvectors descending
        B = Z * E
        ##  L_diag = Tau_scale(B).^2
        L_diag = Tau_scale(B).^2
        L = Diagonal(L_diag)
        cov = E * L * E'

        push!(estimates_sigma, cov)

        ##println("Matrix S", ix, " -- Symmetric=", issymmetric(cov))
        ##display(cov)

        ix += 1
    
        root_cov = sqrtm_symmetric(cov)
        inv_root_cov = inv(root_cov)
        mu = root_cov * median(Z * inv_root_cov, dims=1)[:]
        push!(estimates_mu, mu)
    end

    # Step 3: calculate Mahalanobis distances and get subsets
    best_subsets = HSubset[]

    h = _get_h(model.alpha, X)
    ii = 0
    for (mu, sigma) in zip(estimates_mu, estimates_sigma)
        ii += 1
        dists = mahalanobis_distance(Z, mu, sigma)
        h0 = sortperm(dists)[1:h]
        H = _get_subset(h0, X)
        H = _perform_c_step(model.quan, H, X)
        push!(best_subsets, H)
        ##  @info "Initial subset: ", ii, length(h0), sort(h0), sort(H.indices)
    end

    return best_subsets
end

"""
    CovOgk(;store_precision::Bool=true, assume_centered::Bool=false, location_estimator::Function=median,
        scale_estimator::Function=MAD_scale, n_iterations::Int=2, reweighting::Bool=false,
        reweighting_beta::Float64=0.9)

Implementation of the Orthogonalized Gnanadesikan-Kettenring estimator for location
dispersion proposed in Maronna, R. A., & Zamar, R. H. (2002)

### Keywords:
    store_precision (boolean, optional): whether to store the precision matrix
    assume_centered (boolean, optional): whether the data is already centered
    location_estimator (LocationOrScaleEstimator, optional): function to estimate the
        location of the data, should accept an array like input as first value and a named
        argument axis
    scale_estimator (LocationOrScaleEstimator, optional): function to estimate the scale
        of the data, should accept an array like input as first value and a named argument
        axis
    n_iterations (int, optional): number of iteration for orthogonalization step
    reweighting (boolean, optional): whether to apply reweighting at the end
        (i.e. calculating regular location and covariance after filtering outliers based on
        Mahalanobis distance using OGK estimates)
    reweighting_beta (float, optional): quantile of chi2 distribution to use as cutoff for
        reweighting

### References:
    Maronna, R. A., & Zamar, R. H. (2002).
    Robust Estimates of Location and Dispersion for High-Dimensional Datasets.
    Technometrics, 44(4), 307–317. http://www.jstor.org/stable/1271538

### Examples
```julia
julia> ogk=CovOgk();
julia> fit!(ogk, hbk[:,1:3])
-> Method:  Orthogonalized Gnanadesikan-Kettenring Estimator

Robust estimate of location:
[1.56005, 2.22345, 2.12035]

Robust estimate of covariance:
3×3 Matrix{Float64}:
 3.3575    0.587449  0.699388
 0.587449  2.09268   0.285757
 0.699388  0.285757  2.77527

julia> dd_plot(ogk);
```
"""
mutable struct CovOgk <: RobustCovariance
    store_precision::Bool
    assume_centered::Bool
    location_estimator::Function
    scale_estimator::Function
    n_iterations::Int
    reweighting::Bool
    reweighting_beta::Float64
    nobs::Int
    location_::Union{Vector{Float64}, Nothing}
    covariance_::Union{Matrix{Float64}, Nothing}
    default_location_::Vector{Float64}
    default_covariance_::Matrix{Float64}
    robust_distances_::Vector{Float64}
    mahalanobis_distances_::Vector{Float64}
    X::Union{Matrix{Float64}, Nothing}      # The input data matrix without NAs

    function CovOgk(;
        store_precision::Bool=true,
        assume_centered::Bool=false,
        location_estimator::Function=median,
        scale_estimator::Function=MAD_scale,
        n_iterations::Int=2,
        reweighting::Bool=true,
        reweighting_beta::Float64=0.9
    )
        return new(store_precision, assume_centered, location_estimator, scale_estimator, 
            n_iterations, reweighting, reweighting_beta, 0, [], [;;], [], [;;], [], [], nothing)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", obj::CovOgk)
    println(stdout, "-> Method:  Orthogonalized Gnanadesikan-Kettenring Estimator")
    if isnothing(obj.covariance_) || size(obj.covariance_, 1) == 0
        println()
        println("Model is not fitted yet!")
    else
        println()
        println(io, "Robust estimate of location:")
        println(IOContext(stdout, :compact=>true), location(obj))
        println()
        println(io, "Robust estimate of covariance:")
        Base.show(stdout, mime, covariance(obj))
    end
end

"""
    Calculates location and covariance using the OGK algorithm (Maronna & Zamar, 2002).
"""
function calculate_covariance!(model::CovOgk, X::Matrix{Float64})
    
    p = size(X, 2)
    if p < 2
        error("Needs at least 2 columns!")
    end
    
    Z = copy(X)
    DE = Vector{Matrix{Float64}}()
    
    for _ in 1:model.n_iterations
        s = model.scale_estimator(Z, dims=1)[:]
        D = Diagonal(s)
        Dinv = Diagonal(1.0 ./ s)
        Y = Z * Dinv  # (n x p)
        U = ones(p, p)
        
        ## Compute correlation matrix U
        for i in 2:p
            for j in 1:(i - 1)
                scale_sum = model.scale_estimator(Y[:, i] + Y[:, j])
                scale_diff = model.scale_estimator(Y[:, i] - Y[:, j])
                cor = (scale_sum^2 - scale_diff^2) / (scale_sum^2 + scale_diff^2)
                U[i, j] = U[j, i] = cor
            end
        end
        
        _, E = eigen(U)         # (p x p)
        Z = Y * E               # (n x p)
        push!(DE, D * E)
    end

    cov_X = Diagonal(model.scale_estimator(Z, dims=1)[:] .^ 2)  # (p x p)
    mu_X = model.location_estimator(Z, dims=1)[:]

    for mat in reverse(DE)
        mu_X = mat * mu_X
        cov_X = mat * cov_X * mat'
    end

    if model.reweighting
        mahalanobis = mahalanobis_distance(X, mu_X, cov_X)
        cutoff = sqrt(quantile(Chisq(p), model.reweighting_beta) / quantile(Chisq(p), 0.5)) * median(mahalanobis)
        mask = mahalanobis .< cutoff
        cov_X = cov(X[mask, :], dims=1)
        mu_X = mean(X[mask, :], dims=1)[:]
    end

    model.location_ = mu_X
    model.covariance_ = cov_X

    return cov_X
end

