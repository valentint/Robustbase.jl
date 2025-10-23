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

            ##  Tau_scale matrix version
            tau1 = Tau_scale(Matrix(hbk));
            tau2 = Tau_scale(Matrix(hbk), dims=2);
            tau3 = Tau_scale(hbk[:,1]);
            @test isapprox(tau1, [2.012462, 1.789908, 1.882716, 0.867699], atol=1e-6)
            @test isapprox(tau2[[1,2,3,75]], [8.196243, 8.487519, 8.861269, 0.172557], atol=1e-6)
            @test isapprox(tau3, 2.012462, atol=1e-6)

            ##  Tau_location matrix version
            tau1 = Tau_location(Matrix(hbk));
            tau2 = Tau_location(Matrix(hbk), dims=2);
            tau3 = Tau_location(hbk[:,1]);
            @test isapprox(tau1, [1.555454, 1.879245, 1.741631, -0.0443521], atol=1e-6)
            @test isapprox(tau2[[1,2,3,75]], [15.111434, 15.750955, 15.273159, 0.306762], atol=1e-6)
            @test isapprox(tau3, 1.555454, atol=1e-6)

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
            ## CovClassic
            cc = CovClassic();
            display(cc)
            fit!(cc, hbk[:,1:3])
            display(cc)
            @test isapprox(location(cc), [3.206667, 5.597333, 7.230667], atol=1e-6)
            @test isapprox(covariance(cc), [13.341712 28.469207 41.243982; 28.469207 67.882966 94.665623; 41.243982 94.665623 137.834858], atol=1e-6)

            ## CovMcd
            using Random
            Random.seed!(1234)
            mcd = CovMcd();

            let err = nothing
                try
                  location(mcd)  
                catch err
                end
                @test err isa Exception
                @test sprint(showerror, err) == "Model is not fitted yet!"
            end
            let err = nothing
                try
                  covariance(mcd)  
                catch err
                end
                @test err isa Exception
                @test sprint(showerror, err) == "Model is not fitted yet!"
            end
            let err = nothing
                try
                  correlation(mcd)  
                catch err
                end
                @test err isa Exception
                @test sprint(showerror, err) == "Model is not fitted yet!"
            end

            display(mcd)
            let err = nothing
                try
                    dd_plot(mcd)
                catch err
                end
                @test err isa Exception
                @test sprint(showerror, err) == "Model is not fitted yet!"
            end

            fit!(mcd, hbk[:,1:3]);
            display(mcd)
            dd_plot(mcd)
            @test isapprox(location(mcd), [1.558333,  1.803333,  1.66], atol=1e-6)
            @test isapprox(covariance(mcd), [1.213121    0.0239154  0.1657933; 0.0239154 1.228357 0.195735; 0.165793  0.195735   1.125346], atol=1e-6)
            @test isapprox(correlation(mcd), [1.0 0.019591  0.141896; 0.019591 1.0 0.166480; 0.141896 0.166480 1.0], atol=1e-6)

            ## CovMcd raw estimates only
            mcd = CovMcd(reweighting=false);
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.533333, 2.456410, 1.607692], atol=1e-6)
            @test isapprox(covariance(mcd), [2.817004 0.011009 0.514713; 0.011009 0.890833 0.431404; 0.514713 0.431404 2.156541], atol=1e-6)

            ## Test partitions
            Random.seed!(1234)
            dd = randn(1000, 3)
            mcd=CovMcd(); 
            fit!(mcd, dd);
            @test isapprox(location(mcd), [-0.038419585680912825, -0.03938142931173428, 0.022486439975605767])
            @test isapprox(covariance(mcd), [1.0813375447928146 -0.020273018671691848 0.030248650556126883; -0.020273018671691848 0.961712535610532 0.05609883142210195; 0.030248650556126883 0.05609883142210195 0.9635930203222962])

            ## DetMcd
            mcd = DetMcd();
            display(mcd)
            fit!(mcd, hbk[:,1:3]);
            display(mcd)
            @test isapprox(location(mcd), [1.537705,  1.780327,  1.686885], atol=1e-6)
            @test isapprox(covariance(mcd), [1.220897 0.054737  0.126544; 0.054737 1.2427021 0.151783; 0.126544  0.151783   1.154143], atol=1e-6)

            ## DetMcd raw estimates only
            mcd = DetMcd(reweighting=false);
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.610256, 2.394872, 1.674359], atol=1e-6)
            @test isapprox(covariance(mcd), [2.939714 -0.069041 1.014486; -0.069041 0.913458 0.195346; 1.014486 0.195346 2.143247], atol=1e-6)

            ## CovOgk
            mcd = CovOgk();
            display(mcd)
            fit!(mcd, hbk[:,1:3]);
            display(mcd)
            @test isapprox(location(mcd), [1.513333, 1.808333, 1.701667], atol=1e-6)
            @test isapprox(covariance(mcd), [1.114395 0.093955 0.141672; 0.0939548 1.1231497 0.117444; 0.141672 0.117444 1.0747429], atol=1e-6)

            ## CovOgk raw estimates only
            mcd = CovOgk(reweighting=false);
            fit!(mcd, hbk[:,1:3]);
            @test isapprox(location(mcd), [1.560054, 2.223452, 2.120345], atol=1e-6)
            @test isapprox(covariance(mcd), [3.3574998 0.587449 0.699388; 0.587449 2.092680 0.285757; 0.699388 0.285757 2.775268], atol=1e-6)
        end
    end
 
