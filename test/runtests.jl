using Robustbase
using Test

    @testset "Robustbase.jl" begin

        @testset "Scalers" begin
            X = [1.0, 2.0, 3.0, 100.0, 5.0, NaN, 6.0]

            ## Robustbase.MAD()
            mad_scaler = Robustbase.MAD(can_handle_nan=true);
            fit!(mad_scaler, X, ignore_nan=true);
            @test isapprox(location(mad_scaler), 4.0)
            @test isapprox(scale(mad_scaler), 2.9652)

            fit!(mad_scaler, hbk[:,1])
            @test isapprox(location(mad_scaler), 1.8)
            @test isapprox(scale(mad_scaler), 1.92738, atol=1e-6)

            ## Robustbase.UnivariateMCD()
            mcd_scaler = Robustbase.UnivariateMCD(can_handle_nan=true);
            fit!(mcd_scaler, X, ignore_nan=true);
            @test isapprox(location(mcd_scaler), 3.4)
            @test isapprox(scale(mcd_scaler), 2.958950, atol=1e-6)

            fit!(mcd_scaler, hbk[:,1], ignore_nan=true);
            @test isapprox(location(mcd_scaler), 1.537705, atol=1e-6)
            @test isapprox(scale(mcd_scaler), 1.571673, atol=1e-6)

            ##  Tau scaler
            tau_scaler = Tau(can_handle_nan=true);
            fit!(tau_scaler, X, ignore_nan=true);
            @test isapprox(location(tau_scaler), 3.478855, atol=1e-6)
            @test isapprox(scale(tau_scaler), 3.097441, atol=1e-6)

            fit!(tau_scaler, hbk[:,1])
            @test isapprox(location(tau_scaler), 1.555454, atol=1e-6)
            @test isapprox(scale(tau_scaler), 2.012462, atol=1e-6)

            ##  Qn scaler
            qn_scaler = Qn(can_handle_nan=true);
            fit!(qn_scaler, X, ignore_nan=true);
            @test isapprox(location(qn_scaler), 4.0)                    # the median
            @test isapprox(scale(qn_scaler), 4.06769, atol=1e-6)        # R: 4.075673

            fit!(qn_scaler, hbk[:,1])
            @test isapprox(location(qn_scaler), 1.8)                   # the median
            @test isapprox(scale(qn_scaler), 1.742783, atol=1e-6)      # R: 1.738852

            ##  Tau matrix version
            tau1 = Tau_scale(Matrix(hbk));
            tau2 = Tau_scale(Matrix(hbk), dims=2);
            tau3 = Tau_scale(hbk[:,1]);
            @test isapprox(tau1, [2.012462, 1.789908, 1.882716, 0.867699], atol=1e-6)
            @test isapprox(tau2[[1,2,3,75]], [8.196243, 8.487519, 8.861269, 0.172557], atol=1e-6)
            @test isapprox(tau3, 2.012462, atol=1e-6)

            ##  Qn matrix version
            qn1 = Qn_scale(Matrix(hbk));
            ##  qn2 = Qn_scale(Matrix(hbk), dims=2);    #!! Hangs for ever - FIXME!!!
            qn3 = Qn_scale(hbk[:,1]);
            @test isapprox(qn1, [1.742783, 1.742783, 1.524935, 0.871392], atol=1e-6)
            ##  @test isapprox(qn2[[1,2,3,75]], [8.196243, 8.487519, 8.861269, 0.172557], atol=1e-6)
            @test isapprox(qn3, 1.742784, atol=1e-6)

            ##  MAD matrix version
            mad1 = Robustbase.MAD_scale(Matrix(hbk));
            mad2 = Robustbase.MAD_scale(Matrix(hbk), dims=2);
            mad3 = Robustbase.MAD_scale(hbk[:,1]);
            @test isapprox(mad1, [1.92738, 1.63086, 1.77912, 0.88956], atol=1e-6)
            @test isapprox(mad2[[1,2,3,75]], [7.33887, 8.1543, 7.33887, 0.14826], atol=1e-6)
            @test isapprox(mad3, 1.92738, atol=1e-6)

        end
        @testset "Covariance" begin
            ## CovMcd
            Random.seed!(1234)
            mcd = CovMcd();
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.558333,  1.803333,  1.66], atol=1e-6)
            @test isapprox(covariance(mcd), [1.213121    0.0239154  0.1657933; 0.0239154 1.228357 0.195735; 0.165793  0.195735   1.125346], atol=1e-6)

            ## Test partitions
            Random.seed!(1234)
            dd = randn(1000, 3)
            mcd=CovMcd(); 
            fit!(mcd, dd);
            @test isapprox(location(mcd), [-0.038419585680912825, -0.03938142931173428, 0.022486439975605767])
            @test isapprox(covariance(mcd), [1.0813375447928146 -0.020273018671691848 0.030248650556126883; -0.020273018671691848 0.961712535610532 0.05609883142210195; 0.030248650556126883 0.05609883142210195 0.9635930203222962])

            ## DetMcd
            mcd = DetMcd();
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.537705,  1.780327,  1.686885], atol=1e-6)
            @test isapprox(covariance(mcd), [1.220897 0.054737  0.126544; 0.054737 1.2427021 0.151783; 0.126544  0.151783   1.154143], atol=1e-6)

            ## CovOgk
            mcd = CovOgk();
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.560054, 2.223452, 2.120345], atol=1e-6)
            @test isapprox(covariance(mcd), [3.3574998 0.5874489 0.699388; 0.587449 2.0926801 0.285757; 0.699388 0.285757 2.775268], atol=1e-6)
        end
    end
 
