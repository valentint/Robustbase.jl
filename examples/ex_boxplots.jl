using StatsPlots

##  1. StatsPlots extends Plots.jl with statistical recipes like 
##      boxplot(), violin(), etc. One can control colors, orientation, etc., e.g.
##      boxplot(X, labels=["A", "B", "C"], fillalpha=0.5, orientation=:horizontal)
##

# Example data
using Random
Random.seed!(123)
X = randn(100, 3)  # 100 observations, 3 variables

# Make a boxplot
boxplot(X, labels=["Var1", "Var2", "Var3"], legend=false)

## The previous version will not show labels - to show variable_name
## labels use vectors instead of matrix:

## This does not work either - 
##      two or more boxes are shown for each variable!

# Convert each column to a separate vector for plotting
boxplot(["Var1", "Var2", "Var3"], [X[:,1], X[:,2], X[:,3]],
        legend=false,
        fillalpha=0.5,
        xlabel="Variable",
        ylabel="Value")

        
        
##      Vector-of-vectors + explicit xticks
using StatsPlots, Random
Random.seed!(123)

X = randn(100, 3)
cols = collect(eachcol(X))            # Vector of 3 column vectors
labels = ["Var1", "Var2", "Var3"]

# pass explicit x positions (1,2,3) and the columns; then set xticks
p = boxplot(1:length(cols), cols; legend=false, xlabel="Variable", ylabel="Value", size=(600,400))
xticks!(p, 1:length(cols), labels)    # force the labels on the x axis
display(p)

##      DataFrame (long/tidy) approach — robust and great for layering
##
##      This does not work: ERROR: Cannot convert Symbol to series data for plotting

using DataFrames, StatsPlots, Random
Random.seed!(123)

X = randn(100, 3)
df = DataFrame(X, [:Var1, :Var2, :Var3])
df_long = stack(df, Not([]), variable_name=:variable, value_name=:value)

# this will put one box per level of :variable
p = boxplot(:variable, :value, data=df_long; legend=false, xlabel="Variable", ylabel="Value", size=(600,400))
display(p)

##      DataFrame (long/tidy) approach — robust and great for layering
##      OK - using macro
using DataFrames, StatsPlots, Random
Random.seed!(123)

X = randn(100, 3)
df = DataFrame(X, [:Var1, :Var2, :Var3])
df_long = stack(df, Not([]), variable_name=:variable, value_name=:value)

@df df_long boxplot(:variable, :value;
    legend=false,
    xlabel="Variable",
    ylabel="Value",
    size=(600,400)
)


##      Version without macro
using StatsPlots, DataFrames, Random
Random.seed!(123)

X = randn(100, 3)
df = DataFrame(X, [:Var1, :Var2, :Var3])
df_long = stack(df, Not([]), variable_name=:variable, value_name=:value)

boxplot(df_long.variable, df_long.value;
    legend=false,
    xlabel="Variable",
    ylabel="Value",
    size=(600,400)
)

##============================================
##
##  Minimal example
##
##  This does not work - no labels!
##

using StatsPlots
X = randn(100, 3)
labels = ["Var1", "Var2", "Var3"]
boxplot([X[:,i] for i in 1:3];
    labels=labels,
    legend=false,
    xlabel="Variable",
    ylabel="Value"
)


##==============================================================
##
##  2. Makie gives you full interactivity (zoom, hover) and high-quality 
##      vector exports.
##

using CairoMakie

X = randn(100, 3)

fig = Figure()
ax = Axis(fig[1, 1], xticks=(1:3, ["Var1", "Var2", "Var3"]))
CairoMakie.boxplot!(ax, 1:3, eachcol(X))
fig

##==============================================================
##
##  Using Gadfly.jl (ggplot2-style grammar)
##
using DataFrames, Gadfly

X = randn(100, 3)
df = DataFrame(X, [:Var1, :Var2, :Var3])
df_long = stack(df, Not([]), variable_name=:Variable, value_name=:Value)

plot(df_long, x=:Variable, y=:Value, Geom.boxplot)