using Revise, Pkg
Pkg.activate(".julia/dev/Robustbase")
using Robustbase
pkgversion(Robustbase)

## To run the tests:
##  Change to the root directory of the package, activate the package and run 'test'
cd("C:/Users/valen/.julia/dev/Robustbase")
Pkg.activate(".") 

Pkg.test()


##  To reproduce this CI run locally run the following from the same repository state on julia version 1.12.1:
import Pkg; Pkg.test(;coverage=true, julia_args=["--check-bounds=yes", "--compiled-modules=yes", "--depwarn=yes"], force_latest_compatible_version=false, allow_reresolve=true)

## To build documentation ...
##  Change to the root directory of the package, activate the package and run 'test'
cd("C:/Users/valen/.julia/dev/Robustbase")
using Pkg
Pkg.activate("docs")
include("C:/Users/valen/.julia/dev/Robustbase/docs/make.jl")


## To view the built static documentation:
using LiveServer
servedocs()


##  ==========================================================================
##
##  Testing the scalers
##
Robustbase.MAD_scale(Matrix(hbk))
Robustbase.MAD_scale(Matrix(hbk), dims=2)
Robustbase.MAD_scale(hbk[:,1])

Qn_scale(Matrix(hbk))
##  Qn_scale(Matrix(hbk), dims=2) -     #   this will hang for ever!!!
Qn_scale(hbk[:,1])

Tau_scale(Matrix(hbk))
Tau_scale(Matrix(hbk), dims=2)
Tau_scale(hbk[:,1])

##=============================================================================
X = [1.0, 2.0, 3.0, 100.0, 5.0, NaN, 6.0]
mad_scaler = Robustbase.MAD(can_handle_nan=true)
fit!(mad_scaler, X, ignore_nan=true)

println("Location (Median): ", location(mad_scaler))
println("Scale (MAD): ", scale(mad_scaler))

tau_scaler = Tau(can_handle_nan=true)
fit!(tau_scaler, X, ignore_nan=true)

println("Tau Location: ", location(tau_scaler))
println("Tau Scale: ", scale(tau_scaler))

qn_scaler = Qn(can_handle_nan=true, consistency_correction=false)
fit!(qn_scaler, X, ignore_nan=true)

println("Qn Location: ", location(qn_scaler))
println("Qn Scale: ", scale(qn_scaler))

## ======================================================

##  UnivariateMCD

X = hbk[:,1]
X1 = filter(!isnan, X)
mcd = Robustbase.UnivariateMCD(can_handle_nan=true, alpha=0.75)       # Create UnivariateMCD estimator
Robustbase.set_h_size(mcd.alpha, length(X1))
mcd.h_size=4
Robustbase.get_raw_estimates(X1, mcd.h_size, mcd.consistency_correction)
mcd = fit!(mcd, X, ignore_nan=true)     # Fit the estimator

println("MCD Location: ", location(mcd))
println("MCD Scale: ", scale(mcd))
println("Raw MCD Location: ", mcd.raw_location)
println("Raw MCD Scale: ", mcd.raw_scale)
println("Raw MCD Variance: ", mcd.raw_variance)


##============================================================================
##
##  Testing the scalers by comparing to R on R data sets

using DataFrames
using RCall
using Test

## Load R libraries 
R"library(robustbase)"

function doTestQn(X::Union{Matrix{Float64}, DataFrame}; which::String="Qn")
    if which == "Qn"
        cc=R"apply($X, 2, Qn)"
        scale=Qn_scale(Matrix(X))
    elseif which == "Qn1"
        cc=R"apply($X, 2, Qn)"
        scale=[Robustbase.Qn1(collect(col)) for col in eachcol(X)]
    elseif which == "Qn0"
        cc=R"apply($X, 2, Qn)"
        scale=[Robustbase.Qn0(collect(col)) for col in eachcol(X)]
    elseif which == "Tau"
        cc=R"apply($X, 2, scaleTau2)"
        scale=Tau_scale(Matrix(X))
    else
        error("Undefined scale estimator:", which)       
    end
    @test(isapprox(scale, rcopy(cc)))
end


which = "Qn"
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

## ====================================================

##  Debugging Qn
qn = Qn(finite_correction=true);
z = Vector(hbk[:,1])
fit!(qn, z)

qn0 = Robustbase.Qn0(z)
qn1 = Robustbase.Qn1(z)
qn2 = Robustbase.Qn2(z)
qn0 = Robustbase.Qn0(z, consistency_correction=false, finite_correction=false)
qn1 = Robustbase.Qn1(z, consistency_correction=false, finite_correction=false)
qn2 = Robustbase.Qn2(z, consistency_correction=false, finite_correction=false)


## This will hang for ever ...
qn = Qn();
z = Vector(hbk[33,:])
fit!(qn, z)

##  fit!(qn, [1.5, 3.1, 1.5, -0.6])     # = Vactor(hbk[33,:]), hangs
fit!(qn, [1.5, 3.1, 1.5001, -0.6])      # OK


using DataFrames
using RCall
using Test
using Statistics

## Load R libraries 
R"library(robustbase)"
X = 1.0 * Matrix(rcopy(R"data(machines, package='rrcov'); x=machines[,1:6]"))
##  X = Matrix(rcopy(R"x=heart[,1:2]"))

z = X[:,4]



using Random
Random.seed!(1234)
x = randn(1000);

using BenchmarkTools
using DelimitedFiles

Robustbase.Qn0(x)
Robustbase.Qn1(x)
Qn_scale(x)

bb1 = @benchmark Qn_scale(x) samples=19
bb2 = @benchmark Qn1(x) samples=19
bb3 = @benchmark Qn0(x) samples=19
writedlm("qn-julia.csv", bb1.times, ',')
writedlm("qn1-julia.csv", bb2.times, ',')
writedlm("qn0-julia.csv", bb3.times, ',')



## =======================================================================
##
##  Testing RobustCovariance
##
## =======================================================================

## Test partitioning

using Random
Random.seed!(1234)
dd = randn(10000, 3)
mcd=CovMcd(); 
fit!(mcd, dd)

## ===================================================================
## CovMcd examples

mcd = CovMcd();
fit!(mcd, hbk[:,1:3]);

mcd
location(mcd)
covariance(mcd)
dd_plot(mcd)

## ===================================================================
## CovOgk examples

ogk = CovOgk();     # use the default location and scale estimates (median and mad)
fit!(ogk, hbk[:,1:3])

## Now use the Tau location and scale
ogk = CovOgk(location_estimator=Tau_location, scale_estimator=Tau_scale, reweighting=true);
fit!(ogk, hbk[:,1:3]);
location(ogk)
covariance(ogk)
dd_plot(ogk)


## ===================================================================
## DetMcd examples

using Statistics

X = Matrix(hbk[:,1:3])
Z = (X .- median(X, dims=1)) ./ Qn_scale(X)'
mcd = DetMcd();
fit!(mcd, X)
covariance(mcd)
mcd

##===================================================================
##
##  Testing MCD and DetMcd by comapring to R on many data sets from robustbase
##
using DataFrames
using RCall
using Test

## Load R libraries 
R"library(robustbase)"


function doTest(X::Union{Matrix{Float64}, DataFrame}; reweighting=true, usecc=false, deterministic=false)
    raw_only = !reweighting
    if !deterministic
        cc=R"covMcd($X, raw.only=$raw_only, use.correction=$usecc)"      # returns RObject{VecSxp}
        mcd=CovMcd(reweighting=reweighting); fit!(mcd, X)
    else
        cc=R"covMcd($X, raw.only=$raw_only, use.correction=$usecc, nsamp='deterministic', save.hsets=TRUE, scalefn=Qn)"      # returns RObject{VecSxp}
        mcd=DetMcd(reweighting=reweighting); fit!(mcd, X)
    end
  
    #=
    if !deterministic
        @test(isapprox(mcd.best, rcopy(cc)[Symbol("best")]));
    else
        ii = rcopy(cc)[Symbol("iBest")]
        hSets = rcopy(cc)[Symbol("initHsets")]
        quan = rcopy(cc)[Symbol("quan")]
        best = sort(hSets[Int.(collect(1:quan)), ii[1]])
        @test(isapprox(sort(mcd.best), best));     
    end
    =#

    #=
    @test(isapprox(mcd.best, rcopy(cc)[Symbol("best")]));                   # compare best subset
    @test(isapprox(mcd.crit, rcopy(cc)[Symbol("crit")]));                   # comapre objective function

    =#

    @test(isapprox(mcd.raw_location, rcopy(cc)[Symbol("raw_center")]));     # comapre raw location
    @test(isapprox(mcd.raw_covariance, rcopy(cc)[Symbol("raw_cov")]));      # comapre raw covariance

    @test(isapprox(location(mcd), rcopy(cc)[Symbol("center")]));     # comapre location
    @test(isapprox(covariance(mcd), rcopy(cc)[Symbol("cov")]));      # comapre covariance

end

usecc = false
reweighting = true
deterministic = true

## Load data set
@info "Testing data set Animals"
X = rcopy(R"data(Animals, package='MASS'); x=Animals"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set aircraft"
X = rcopy(R"x=aircraft[,1:4]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set heart"
X = rcopy(R"x=heart[,1:2]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set bushfire: Integer matrix"
X = 1.0 * Matrix(rcopy(R"x=bushfire")); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set coleman"
X = rcopy(R"x=coleman[,1:5]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set delivary: integer matrix"
X = 1.0 * Matrix(rcopy(R"x=delivery[,1:2]")); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set maryo"
X = rcopy(R"data(maryo, package='rrcov'); x=maryo"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set phosphor"
X = rcopy(R"x=phosphor[,1:2]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set salinity"
X = rcopy(R"x=salinity[,1:3]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set wood"
X = rcopy(R"x=wood[,1:5]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

##================== With set seed - differences in mcd.best
@info "Testing data set starsCYG"
X = rcopy(R"set.seed(1234);x=starsCYG"); import Random; Random.seed!(1234); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set education"
X = 1.0 * Matrix(rcopy(R"set.seed(1234);x=education[,3:5]")); import Random; Random.seed!(1234); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

##================== With set seed
@info "Testing data set hbk"
X = rcopy(R"set.seed(1234); x=hbk[,1:3]"); import Random; Random.seed!(1234); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set hemophilia"
X = rcopy(R"set.seed(1234); data(hemophilia, package='rrcov'); x=hemophilia[,1:2]"); import Random; Random.seed!(9999); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set un86"
X = rcopy(R"set.seed(1234); data(un86, package='rrcov'); x=un86"); import Random; Random.seed!(9999); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set rice"
X = rcopy(R"set.seed(5678); data(rice, package='rrcov'); x=rice[,1:5]"); import Random; Random.seed!(9999); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)


##================== Sometimes different

##  ........


##============== ERRORS
@info "Testing data set milk"
X = rcopy(R"set.seed(1234); x=milk"); import Random; Random.seed!(5678); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set machines - integer matrix"
X = 1.0 * Matrix(rcopy(R"data(machines, package='rrcov'); x=machines[,1:6]")); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set ionosphere"
X = rcopy(R"data(ionosphere, package='rrcov'); x=ionosphere[,3:34]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)


##=================== Missing data
@info "Testing data set airmay - has missing data"
# !!! X = rcopy(R"x=airmay[,1:3]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)

@info "Testing data set fish: missing data"
# !!! X = rcopy(R"data(fish, package='rrcov'); x=fish[,1:6]"); doTest(X, reweighting=reweighting, usecc=usecc, deterministic=deterministic)


## Test alpha=1
mcd = CovMcd(alpha=1);
fit!(mcd, hbk[:,1:3]);
mcd

mcd = DetMcd(alpha=1);
fit!(mcd, hbk[:,1:3]);
mcd

## Test p=1
mcd = CovMcd();
fit!(mcd, hbk[:,1]);
mcd

mcd = DetMcd();
fit!(mcd, hbk[:,1]);
mcd

##================================================
using DataFrames
using RCall

## Load R libraries 
R"library(cellWise)"

X = rcopy(R"data('data_philips', package='cellWise'); x=data_philips");

mcd=CovMcd()
fit!(mcd, X)

##================================================
##
##  Test all Plots
##

cc = CovClassic()
mcd = CovMcd()
dmcd = DetMcd()
ogk = CovOgk()

fit!(cc, hbk[:,1:3])
try
    dd_plot(cc)
catch err
end
qq_plot(cc)
distance_plot(cc)
tolellipse_plot(cc)

fit!(mcd, hbk[:,1:3])
dd_plot(mcd)
qq_plot(mcd)
distance_plot(mcd)
tolellipse_plot(mcd)

fit!(dmcd, hbk[:,1:3])
dd_plot(dmcd)
qq_plot(dmcd)
distance_plot(dmcd)
tolellipse_plot(dmcd)

fit!(ogk, hbk[:,1:3])
dd_plot(ogk)
qq_plot(ogk)
distance_plot(ogk)
tolellipse_plot(ogk)
