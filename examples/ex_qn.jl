using Robustbase 

##  Debugging Qn
qn = Qn(finite_correction=true);
z = Vector(hbk[:,1])
fit!(qn, z)


## Naive O(n^2) implementation
qn0 = Qn0(z)
qn0 = Qn0(z, consistency_correction=false, finite_correction=false)

## Efficient O(n log n) implementation, the Rousseeuw & Croux algorithm
qn1 = Qn1(z)
qn1 = Qn1(z, consistency_correction=false, finite_correction=false)

## Translation of the Python function
qn2 = Qn2(z)
qn2 = Qn2(z, consistency_correction=false, finite_correction=false)


## This will hang for ever ...
z = Vector(hbk[33,:])
##  Qn2(z)

##  Qn2([1.5, 3.1, 1.5, -0.6])     # = Vector(hbk[33,:]), hangs
Qn2([1.5, 3.1, 1.5001, -0.6])      # OK

## Benchmark ... =====================================================
using Random
Random.seed!(1234)
x = randn(1000);

using BenchmarkTools
using DelimitedFiles

Qn0(x)
Qn1(x)
Qn2(x)
Qn_scale(x)

bb1 = @benchmark Qn_scale(x) samples=19
bb2 = @benchmark Qn0(x) samples=19
bb3 = @benchmark Qn1(x) samples=19
bb4 = @benchmark Qn2(x) samples=19

writedlm("qn-julia.csv", bb1.times, ',')
writedlm("qn0-julia.csv", bb2.times, ',')
writedlm("qn1-julia.csv", bb3.times, ',')
writedlm("qn2-julia.csv", bb4.times, ',')

## Plot it
using StatsPlots
X = [reshape(bb1.times/1e9, 1, :); reshape(bb2.times/1e9, 1, :); 
        reshape(bb3.times/1e9, 1, :); reshape(bb4.times/1e9, 1, :)]'

boxplot(X, xticks=(1:4, ["Qn{jl}", "Qn0", "Qn1", "Qn{py}"]), legend=false)

## Compare to R ====================================================
using DataFrames
using RCall
using Test

## Load R libraries 
R"library(robustbase)"

function doTestQn(X::Union{Matrix{Float64}, DataFrame}; which::String="Qn", atol=0)
    if which == "Qn"
        cc=R"apply($X, 2, Qn)"
        scale=Qn_scale(Matrix(X))
    elseif which == "Qn1"
        cc=R"apply($X, 2, Qn)"
        scale=[Qn1(collect(col)) for col in eachcol(X)]
    elseif which == "Qn0"
        cc=R"apply($X, 2, Qn)"
        scale=[Qn0(collect(col)) for col in eachcol(X)]
    elseif which == "Qn2"
        cc=R"apply($X, 2, Qn)"
        scale=[Qn2(collect(col)) for col in eachcol(X)]
    elseif which == "Tau"
        cc=R"apply($X, 2, scaleTau2)"
        scale=Tau_scale(Matrix(X))
    else
        error("Undefined scale estimator:", which)       
    end
    @test(isapprox(scale, rcopy(cc), atol=atol))
end


which = "Qn2"
## Load data set
@info "Testing data set Animals"
X = rcopy(R"data(Animals, package='MASS'); x=Animals"); doTestQn(X, which=which)

@info "Testing data set aircraft"
X = rcopy(R"x=aircraft[,1:4]"); doTestQn(X, which=which)

@info "Testing data set heart"
X = rcopy(R"x=heart[,1:2]"); doTestQn(X, which=which)

@info "Testing data set bushfire: Integer matrix"
X = 1.0 * Matrix(rcopy(R"x=bushfire")); doTestQn(X, which=which)

@info "Testing data set coleman"
X = rcopy(R"x=coleman[,1:5]"); doTestQn(X, which=which)

@info "Testing data set delivary: integer matrix"
X = 1.0 * Matrix(rcopy(R"x=delivery[,1:2]")); doTestQn(X, which=which)

@info "Testing data set maryo"
X = rcopy(R"data(maryo, package='rrcov'); x=maryo"); doTestQn(X, which=which)

@info "Testing data set phosphor"
X = rcopy(R"x=phosphor[,1:2]"); doTestQn(X, which=which)

@info "Testing data set salinity"
X = rcopy(R"x=salinity[,1:3]"); doTestQn(X, which=which)

@info "Testing data set wood"
X = rcopy(R"x=wood[,1:5]"); doTestQn(X, which=which)

@info "Testing data set starsCYG"
X = rcopy(R"x=starsCYG"); doTestQn(X, which=which)

@info "Testing data set education"
X = 1.0 * Matrix(rcopy(R"x=education[,3:5]")); doTestQn(X, which=which)

@info "Testing data set hbk"
X = rcopy(R"x=hbk[,1:3]"); doTestQn(X, which=which, atol=1e-7)

@info "Testing data set hemophilia"
X = rcopy(R"data(hemophilia, package='rrcov'); x=hemophilia[,1:2]"); doTestQn(X, which=which)

@info "Testing data set un86"
X = rcopy(R"data(un86, package='rrcov'); x=un86"); doTestQn(X, which=which)

@info "Testing data set rice"
X = rcopy(R"data(rice, package='rrcov'); x=rice[,1:5]"); doTestQn(X, which=which)

@info "Testing data set milk"
X = rcopy(R"x=milk"); doTestQn(X, which=which, atol=1e-6)

if which != "Qn2"
    @info "Testing data set machines - integer matrix"
    X = 1.0 * Matrix(rcopy(R"data(machines, package='rrcov'); x=machines[,1:6]")); doTestQn(X, which=which)
end

@info "Testing data set ionosphere"
X = rcopy(R"data(ionosphere, package='rrcov'); x=ionosphere[,3:34]"); doTestQn(X, which=which)

"""
    A simple version of Qn() -- O(n^2) in both time and memory — 
    good for small to moderate n, easy to understand
"""




function Qn0(x::AbstractVector{<:Real}; consistency_correction::Bool=true, finite_correction::Bool=true)
    x = sort(collect(x))
    n = length(x)
    if n == 0
        return NaN
    elseif n == 1
        return 0.0
    end

    k = binomial(div(n, 2) + 1, 2)

    # pairwise differences (lower triangle only)
    diffs = Float64[]
    for i in 2:n
        for j in 1:(i - 1)
            push!(diffs, x[i] - x[j])
        end
    end

    sort!(diffs)

    q_raw = diffs[k]

    cn = cc = 1.0
    ## finite-sample correction factor
    if finite_correction
        cn = Robustbase.get_small_sample_cn(n)
    end

    ## conistency correction factor
    if consistency_correction
        cc = 2.21914
    end

    return cc * q_raw / cn
end

"""
    Efficient O(n log n) version of Qn (Rousseeuw & Croux algorithm) - 
    this version will be used in Robustbase.

"""
function Qn1(x::AbstractVector{<:Real}; consistency_correction::Bool=true, finite_correction::Bool=true)
    x = sort(collect(x))
    n = length(x)
    if n == 0
        return NaN
    elseif n == 1
        return 0.0
    end

    """
        k is typically half of n, specifying the "quantile", i.e., rather the order 
        statistic that Qn() should return; for the Qn() proper, this has been hard 
        wired to choose(n%/%2 +1, 2), i.e., floor(n/2) + 1. Choosing a large k is 
        less robust but allows to get non-zero results in case the default Qn() is zero.

        If k is the default, the consistency correction constant is 2.219144.
    """
    k = binomial(div(n, 2) + 1, 2)

    ## count how many |x_i - x_j| ≤ d
    function count_pairs_leq(d)
        count = 0
        j = 1
        for i in 1:n
            while j ≤ n && x[j] - x[i] ≤ d
                j += 1
            end
            count += j - i - 1
        end
        return count
    end

    ## binary search for smallest distance where count ≥ k
    lo = 0.0
    hi = x[end] - x[1]
    for _ in 1:60
        mid = (lo + hi) / 2
        if count_pairs_leq(mid) < k
            lo = mid
        else
            hi = mid
        end
    end
    q_raw = hi

    cn = cc = 1.0
    ## finite-sample correction factor
    if finite_correction
        cn = Robustbase.get_small_sample_cn(n)
    end

    ## conistency correction factor
    if consistency_correction
        cc = 2.21914
    end

    return cc * q_raw / cn
end

"""
    weighted_median(X::Vector{Float64}, weights::Vector{Float64})

    Computes a weighted median.

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
    This is a translation from the Python version of Qn.
    Slower than Qn1() above and buggy - there are cases 
    when it hangs indefinitely, also the result is slightly different from R.
"""
function Qn2(X::AbstractVector{<:Real}; consistency_correction::Bool=true, finite_correction::Bool=true)
    n = length(X)
    h = div(n, 2) + 1
    
    """
        k is typically half of n, specifying the "quantile", i.e., rather the order 
        statistic that Qn() should return; for the Qn() proper, this has been hard 
        wired to choose(n%/%2 +1, 2), i.e., floor(n/2) + 1. Choosing a large k is 
        less robust but allows to get non-zero results in case the default Qn() is zero.

        If k is the default, the consistency correction constant is 2.219144.
    """
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
        trial = weighted_median(work .+ 0.0, weight .+ 0.0)
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

    if finite_correction
        Qn_val /= Robustbase.get_small_sample_cn(n)
    end
    if consistency_correction
        Qn_val *= 2.21914
    end

     return Qn_val
end


