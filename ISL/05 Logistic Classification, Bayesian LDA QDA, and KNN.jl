# ## Stock market data

# Let's load the usual packages and the data
using MLJ
import RDatasets: dataset #import does not bring the package and its fucntions in the namespace
import DataFrames: DataFrame, describe, select, Not
import StatsBase: countmap, cor, var

smarket = dataset("ISLR", "Smarket")
@show size(smarket)
@show names(smarket)

# Since we often  want  to only show a few significant digits for the metrics etc, let's introduce a very simple function  that does that:
r3(x) = round(x, sigdigits=3)
r3(pi)
ϵ = 0.1

# Let's get a description too
println(describe(smarket, :mean, :std, :eltype))

# The target variable is `:Direction`:

y = smarket.Direction
X = select(smarket, Not(:Direction))

# We can compute all the pairwise correlations; we use `Matrix` so that the dataframe entries are considered as one matrix of numbers with the same type (otherwise `cor` won't work):

using Plots
# using Plots: text

function plotter(cr::Matrix{Float64}, cols::Vector{Symbol})::Nothing
    (n, m) = size(cr)

    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="Pearson Correlation Coefficients")

    annotate!([(j, i, text(round(cr[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])

    savefig("./pearson_correlations.png")
    return nothing
end

cm = X |> Matrix |> cor
plotter(round.(cm, sigdigits=1), Symbol.(names(X)))

# Let's see what the `:Volume` feature looks like:

using Plots

begin
    plot(X.Volume, size=(800, 600), linewidth=2, legend=false)
    xlabel!("Tick number")
    ylabel!("Volume")
end


# ### Logistic Regression

# We will now try to train models; the target `:Direction` has two classes: `Up` and `Down`; it needs to be interpreted as a categorical object, and we will mark it as a _ordered factor_ to specify that 'Up' is positive and 'Down' negative (for the confusion matrix later):

y = coerce(y, OrderedFactor)
levels(y)

# Note that in this case the default order comes from the lexicographic order which happens  to map  to  our intuition since `D`  comes before `U`.

cm = countmap(y)
categories, vals = collect(keys(cm)), collect(values(cm))
Plots.bar(categories, vals, title="Bar Chart Example", legend=false)
ylabel!("Number of occurrences")

# Seems pretty balanced.

# Let's now try fitting a simple logistic classifier (aka logistic regression) not using `:Year` and `:Today`:

LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels
X2 = select(X, Not([:Year, :Today]))
classif = machine(LogisticClassifier(), X2, y)

# Let's fit it to the data and try to reproduce the output:

fit!(classif)
ŷ = MLJ.predict(classif, X2)
ŷ[1:3]

# Note that here the `ŷ` are _scores_.
# We can recover the average cross-entropy loss:
cross_entropy(ŷ, y) |> r3

# in order to recover the class, we could use the mode and compare the misclassification rate:

ŷ = predict_mode(classif, X2)
misclassification_rate(ŷ, y) |> r3

# Well that's not fantastic...
#
# Let's visualise how we're doing building a confusion matrix,
# first is predicted, second is truth:

@show cm = confusion_matrix(ŷ, y)

# We can then compute the accuracy or precision, etc. easily for instance:

@show false_positive(cm)
@show accuracy(ŷ, y) |> r3
@show accuracy(cm) |> r3  # same thing
@show ppv(cm) |> r3   # a.k.a. precision
@show recall(cm) |> r3
@show f1score(ŷ, y) |> r3
@show fpr(ŷ, y) |> r3

# Let's now train on the data before 2005 and use it to predict on the rest.
# Let's find the row indices for which the condition holds

train = 1:findlast(X.Year .< 2005)
test = last(train)+1:length(y)

# We can now just re-fit the machine that we've already defined just on those rows and predict on the test:

fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Well, that's not very good...
# Let's retrain a machine using only `:Lag1` and `:Lag2`:

X3 = select(X2, [:Lag1, :Lag2])
classif = machine(LogisticClassifier(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Interesting... it has higher accuracy than the model with more features! This could be investigated further by increasing the regularisation parameter but we'll leave that aside for now.
#
# We can use a trained machine to predict on new data:

Xnew = (Lag1=[1.2, 1.5], Lag2=[1.1, -0.8])
ŷ = MLJ.predict(classif, Xnew)
ŷ |> println

# **Note**: when specifying data, we used a simple `NamedTuple`; we could also have defined a dataframe or any other compatible tabular container.
# Note also that we retrieved the raw predictions here i.e.: a score for each class; we could have used `predict_mode` or indeed

mode.(ŷ)


#HW TODO - Evaluate your LogisticClassifier using 10-folds

# ### LDA

# Let's do a similar thing but with a LDA model this time:

BayesianLDA = @load BayesianLDA pkg = MultivariateStats

classif = machine(BayesianLDA(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3

# Note: `BayesianLDA` is LDA using a multivariate normal model for each class with a default prior inferred from the proportions for each class in the training data.
# You can also use the bare `LDA` model which does not make these assumptions and allows using a different metric in the transformed space, see the docs for details.

LDA = @load LDA pkg = MultivariateStats
using Distances

classif = machine(LDA(dist=CosineDist()), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3

# ### QDA
#
# Bayesian QDA is available via ScikitLearn:

BayesianQDA = @load BayesianQDA pkg = MLJScikitLearnInterface

# Using it is done in much the same way as before:

classif = machine(BayesianQDA(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3


using StatsPlots
begin
    density(X3.Lag1 - X3.Lag2, label="Lag1 - Lag2", linewidth=2)
    density!(X3.Lag2 - X3.Lag1, label="Lag2 - Lag1", linewidth=2)
    xlabel!("Difference")
    ylabel!("Density")
    title!("Density Plot of Lag Differences")
end

begin
    marginalkde(X3.Lag1 - X3.Lag2, X3.Lag2 - X3.Lag1, label="Marginal KDE", linewidth=2)
    xlabel!("Lag1 - Lag2")
    ylabel!("Lag2 - Lag1")
    title!("KDE Plot of Lag Differences")
end


# ### KNN

# We can use K-Nearest Neighbors models via the [`NearestNeighbors`](https://github.com/KristofferC/NearestNeighbors.jl) package:

KNNClassifier = @load KNNClassifier

knnc = KNNClassifier(K=2)
classif = machine(knnc, X2, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Pretty bad... let's try with three neighbors

knnc.K = 3
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

knnc.K = 4
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# A bit better but not hugely so.

knnc.K = 10
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

#Worse