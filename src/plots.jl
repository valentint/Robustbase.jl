#=
    This new version uses the StatsPlots recipes, particularly, 
    creates a light data frame and uses the macro @df to plot 
    the scatterplot. The annotations of the outliers are given as 
    keword.
=#
"""
        dd_plot(model::RobustCovariance; chi2_percentile::Float64=0.975, 
            id_n=nothing, figsize=(400, 400))

Produce distance-distance plot for a robust estiamtor of location and covariance. 
The robust mahalanobis distances are plotted against the classical mahalanobis distances. A horizontal and vertical line identify the threshold
related to a percentile of the chi2 distribution (```chi2_percentile```).
The ```id_n``` observations with largest robust distances are annotated by labels. 

# Attrributes
- ```model```: a ```RobustCovariance``` estimator.

# Keywords
- ```chi2_percentile```: specifies the cutoff for outlier detection, a quantile of the chi2 distribution. Default is ```chi2_percentile=0.975```.
- ```id_n```: number of observations to annotate. By default these are the observations with robust mahalanobis distances larger than a thershold given by ```chi2-percentile```.
- ```figsize```: size of the plot, default is ```figsize=(400, 400)```.

# Returns
The current plot.
"""
function dd_plot(model::RobustCovariance; chi2_percentile::Float64=0.975, id_n=nothing, figsize=(400, 400))

    _fitted_covariance(model)
    dist = distance(model)
    n = length(dist)
    p = length(location(model))
    threshold = sqrt(quantile(Chisq(p), chi2_percentile))

    ## Create a tidy DataFrame for StatsPlots
    df = DataFrame(
        mahalanobis = model.mahalanobis_distances_,
        robust      = dist,
        index       = 1:n
    )

    max_x = maximum([maximum(df.mahalanobis), threshold*1.1])
    max_y = maximum([maximum(df.robust), threshold*1.1])

    # Determine number of labels to show
    if isnothing(id_n)
        id_n = length(findall(df.robust .>= threshold))
        ##  println("id_n: ", id_n)
    end

    # Prepare annotations for the top id_n points
    top_indices = sortperm(df.robust, rev=true)[1:min(id_n, nrow(df))]
    off = 0.045 * max_x
    annotations = [(df.mahalanobis[i]-off, df.robust[i], text(i, :red, 6, :left)) for i in top_indices]

    # Use StatsPlots scatter recipe
    p = @df df scatter(
        :mahalanobis, :robust;
        xlabel="Non-robust distance",
        ylabel="Robust distance",
        title="Distance-Distance Plot",
        legend=false,
        size=figsize,
        color=:skyblue,
        markerstrokewidth=0.5,
        annotations=annotations
    )

    # Add threshold lines and diagonal
    hline!(p, [threshold], color=:gray, linestyle=:dash)
    vline!(p, [threshold], color=:gray, linestyle=:dash)
    plot!(p, [0, max_x], [0, max_y], color=:gray, linestyle=:dashdot)

    return current()
end

"""
    qq_plot(model::CovarianceEstimator; chi2_percentile::Float64=0.975, 
        id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

Produce chi2 Q-Q plot for a robust or classical estiamtor of multivariate location and covariance. 
The sample quantiles (sorted (robust) mahalanobis distances) are plotted against the squared root 
of the quantiles of the chi2 distribution with ```p``` degrees of freedom (where ```p``` is the 
dimension of the data matrix). If the two distributions are identical, the Q–Q plot follows 
the 45° line ```y = x``` (the dotted line).
The ```id_n``` observations with largest distances are annotated by labels. 

# Attrributes
- ```model```: a ```CovarianceEstimate``` estimator.

# Keywords
- ```chi2_percentile```: specifies the cutoff for outlier detection, a quantile of the chi2 distribution. Default is ```chi2_percentile=0.975```.
- ```id_n```: number of observations to annotate. By default these are the observations with robust mahalanobis distances larger than a thershold given by ```chi2-percentile```.
- ```figsize```: size of the plot, default is ```figsize=(400, 400)```.

# Returns
The current plot.
"""
function qq_plot(model::CovarianceEstimator; chi2_percentile::Float64=0.975, id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

    _fitted_covariance(model)
    dist = distance(model)
    n = length(dist)
    p = length(location(model))
    robust = model isa RobustCovariance;
    threshold = sqrt(quantile(Chisq(p), chi2_percentile))

    # Create a tidy DataFrame for StatsPlots
    chi2 = Chisq(p) 
    qq = sqrt.(quantile(chi2, ((1:n) .- 1/3) ./ (n+1/3)))
    idx = sortperm(dist)
    df = DataFrame(
        qq = qq,
        dist = dist,
        index = idx
    )

    max_x = maximum([maximum(df.qq), threshold*1.1])
    max_y = maximum([maximum(df.dist), threshold*1.1])

    # Determine number of labels to show
    if isnothing(id_n)
        id_n = length(findall(df.dist .>= threshold))
    end

    # Prepare annotations for the top id_n points
    ##  top_indices = sortperm(df.dist, rev=true)[1:min(id_n, nrow(df))]
    top_indices = (n-id_n+1):n
    off = 0.045 * max_x
    annotations = [(df.qq[i]-off, df.dist[df.index[i]], text(df.index[i], :red, 6, :left)) for i in top_indices]

    xlab = "Square root of the quantiles of the " * "\$ \\chi^2\$" * " distribution"
    ylab = if(robust) "Robust distance" else "Mahalanobis distance" end
    main = "\$\\chi^2\$" * " QQ-Plot"

    qqplot(df.qq, df.dist, 
        annotations=annotations, 
        xlabel=xlab,
        ylabel=ylab,
        title=main,
        guidefontsize=9,
        color=:skyblue,
        linecolor=:red,
        size=figsize
    )
#=
    # Use StatsPlots scatter recipe
    p = @df df scatter(
        :qq, :dist[:index];
        xlabel=xlab,
        ylabel=ylab,
        title=main,
        legend=false,
        size=figsize,
        color=:skyblue,
        markerstrokewidth=0.5,
        annotations=annotations
    )
=#    
    return current()    
end

"""
    distance_plot(model::CovarianceEstimator; chi2_percentile::Float64=0.975, 
        id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

Produce distance plot for a robust or classical estiamtor of multivariate location and covariance. 
The (robust) mahalanobis distances are plotted against the index of the observations. 
A horizontal line identifies the threshold related to a percentile of the chi2 distribution (```chi2_percentile```). The ```id_n``` observations with largest distances are annotated by labels. 

# Attrributes
- ```model```: a ```CovarianceEstimate``` estimator.

# Keywords
- ```chi2_percentile```: specifies the cutoff for outlier detection, a quantile of the chi2 distribution. Default is ```chi2_percentile=0.975```.
- ```id_n```: number of observations to annotate. By default these are the observations with robust mahalanobis distances larger than a thershold given by ```chi2-percentile```.
- ```figsize```: size of the plot, default is ```figsize=(400, 400)```.

# Returns
The current plot.
"""
function distance_plot(model::CovarianceEstimator; chi2_percentile::Float64=0.975, id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

    _fitted_covariance(model)
    dist = distance(model)
    n = length(dist)
    p = length(location(model))
    robust = model isa RobustCovariance;
    threshold = sqrt(quantile(Chisq(p), chi2_percentile))

    ## Create a tidy DataFrame for StatsPlots
    df = DataFrame(
        distance = dist,
        index    = 1:length(dist)
    )

    max_x = maximum([maximum(df.index), threshold*1.1])
    max_y = maximum([maximum(df.distance), threshold*1.1])

    # Determine number of labels to show
    if isnothing(id_n)
        id_n = length(findall(df.distance .>= threshold))
        ##  println("id_n: ", id_n)
    end

    # Prepare annotations for the top id_n points
    top_indices = sortperm(df.distance, rev=true)[1:min(id_n, n)]
    off = 0.045 * max_x
    annotations = [(df.index[i]-off, df.distance[i], text(i, :red, 6, :left)) for i in top_indices]
    xlab = "Observation index"
    ylab = if(robust) "Robust distance" else "Mahalanobis distance" end
    main = "Distance Plot"

    # Use StatsPlots scatter recipe
    p = @df df scatter(
        :index, :distance;
        xlabel=xlab,
        ylabel=ylab,
        title=main,
        legend=false,
        size=figsize,
        color=:skyblue,
        markerstrokewidth=0.5,
        annotations=annotations
    )

    # Add threshold lines and diagonal
    hline!(p, [threshold], color=:gray, linestyle=:dash)
    
    return current()

end

"""
    tolellipse_plot(model::CovarianceEstimator; select=[1, 2], 
    classic::Bool=true, chi2_percentile::Float64=0.975, 
    id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

Produce tolerance ellipse plot for a robust or classical estiamtor of multivariate location and covariance. 
A scatter plot of two selected columns of the data matrix are plotted and tolerance ellipse(s) with confidence ```chi2_percentile``` are overlayed.
In the case of a robust estiamtor both the elipses based on the robust and classical estimates are shown while in the case of a classical estimator only the classical ellipse is shown.
The ```id_n``` observations with largest distances are annotated by labels. 

# Attrributes
- ```model```: a ```CovarianceEstimate``` estimator.

# Keywords
- ```select```: A vector of two elements with the indexes of the two columns to be plotted. Default is ```select=[1, 2]```.
- ```classic```: Whether to print the classical tolerance ellipse (in case of a robust estimator)
- ```chi2_percentile```: specifies the cutoff for outlier detection, a quantile of the chi2 distribution. Default is ```chi2_percentile=0.975```.
- ```id_n```: number of observations to annotate. By default these are the observations with robust mahalanobis distances larger than a thershold given by ```chi2-percentile```.
- ```figsize```: size of the plot, default is ```figsize=(400, 400)```.

# Returns
The current plot.
"""
function tolellipse_plot(model::CovarianceEstimator; select=[1, 2], classic::Bool=true, chi2_percentile::Float64=0.975, id_n::Union{Int64, Nothing}=nothing, figsize=(400, 400))

    _fitted_covariance(model)
    if !hasproperty(model, :X)
        error("Model has no 'data' property!")
    end
    if isnothing(model.X) || isempty(model.X)
        error("No data provided!")
    end

    dist = distance(model)
    n = size(model.X, 1)
    p = length(location(model))
    robust = model isa RobustCovariance;
    threshold = sqrt(quantile(Chisq(p), chi2_percentile))
    names = ["X$i" for i=1:size(model.X, 2)]

    if(p == 1)
        error("Tolerance ellipse plot not possible for univariate data!")
    end
    if(length(select) != 2 || select[1] < 1 || select[1] > p || select[2] < 1 || select[2] > p)
        error("Invalid columns selected: both should be greater than 1 and less than ", p)
    end
    if(select[1] == select[2])
        error("Identical columns selected!")
    end
    mu = location(model)[select]
    sigma = covariance(model)[select, select]
    if(robust)
        mu = model.default_location_[select]
        sigma = model.default_covariance_[select,select]
        rmu = location(model)[select]
        rsigma = covariance(model)[select,select]
    end

    ## Create a tidy DataFrame for StatsPlots
    df = DataFrame(
        X1 = model.X[:, select[1]],
        X2 = model.X[:, select[2]],
        distance = dist,
        index    = 1:n
    )

    max_x = maximum([maximum(df.X1), threshold*1.1])
    max_y = maximum([maximum(df.X2), threshold*1.1])

    # Determine number of labels to show
    if isnothing(id_n)
        id_n = length(findall(df.distance .>= threshold))
        ##  println("id_n: ", id_n)
    end

    # Prepare annotations for the top id_n points
    top_indices = sortperm(df.distance, rev=true)[1:min(id_n, n)]
    off = 0.045 * max_x
    annotations = [(df.X1[i]-off, df.X2[i], text(i, :red, 6, :left)) for i in top_indices]
    xlab = names[select[1]]
    ylab = names[select[2]]
    main = "Tolerance ellipse (" * "$(100*chi2_percentile)%)"

    # Use StatsPlots scatter recipe
    p = @df df scatter(
        :X1, :X2;
        xlabel=xlab,
        ylabel=ylab,
        title=main,
        label="",
        size=figsize,
        color=:skyblue,
        markerstrokewidth=0.5,
        annotations=annotations
    )

    if(robust)
        plot!(getellipsepoints(rmu, rsigma, chi2_percentile)..., label="robust", color=:red)
    end
    if(!robust || classic)
        plot!(getellipsepoints(mu, sigma, chi2_percentile)..., label="classical", color=:blue)
    end

    return current()
end

## Generate ellipse with center radius 1 and radius 2
function getellipsepoints(cx, cy, rx, ry, θ)
	t = range(0, 2*pi, length=100)
	ellipse_x_r = @. rx * cos(t)
	ellipse_y_r = @. ry * sin(t)
	R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
	r_ellipse = [ellipse_x_r ellipse_y_r] * R
	x = @. cx + r_ellipse[:,1]
	y = @. cy + r_ellipse[:,2]
	(x,y)
end

## Generate ellipse with location, covariance and tollerance
function getellipsepoints(mu, sigma, confidence=0.975)
	quant = quantile(Chisq(2), confidence) |> sqrt
	cx = mu[1]
	cy =  mu[2]
	
	egvs = eigvals(sigma)
	if egvs[1] > egvs[2]
		idxmax = 1
		largestegv = egvs[1]
		smallesttegv = egvs[2]
	else
		idxmax = 2
		largestegv = egvs[2]
		smallesttegv = egvs[1]
	end

	rx = quant*sqrt(largestegv)
	ry = quant*sqrt(smallesttegv)
	
	eigvecmax = eigvecs(sigma)[:,idxmax]
	θ = atan(eigvecmax[2]/eigvecmax[1])
 	if θ < 0
		θ += 2*π
	end

	getellipsepoints(cx, cy, rx, ry, θ)
end