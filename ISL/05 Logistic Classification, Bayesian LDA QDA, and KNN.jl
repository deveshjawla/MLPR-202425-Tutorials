

# ## Stock market data


#
# Let's load the usual packages and the data

using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, describe, select, Not
import StatsBase: countmap, cor, var
using PrettyPrinting

smarket = dataset("ISLR", "Smarket")
@show size(smarket)
@show names(smarket)

# Since we often  want  to only show a few significant digits for the metrics etc, let's introduce a very simple function  that does that:

r3(x) = round(x, sigdigits=3)
r3(pi)

# Let's get a description too

describe(smarket, :mean, :std, :eltype)

# The target variable is `:Direction`:

y = smarket.Direction
X = select(smarket, Not(:Direction));

# We can compute all the pairwise correlations; we use `Matrix` so that the dataframe entries are considered as one matrix of numbers with the same type (otherwise `cor` won't work):

cm = X |> Matrix |> cor
round.(cm, sigdigits=1)

# Let's see what the `:Volume` feature looks like:

using Plots

plot(X.Volume, size=(800,600), linewidth=2, legend=false)
xlabel!("Tick number")
ylabel!("Volume")



# ### Logistic Regression


#
# We will now try to train models; the target `:Direction` has two classes: `Up` and `Down`; it needs to be interpreted as a categorical object, and we will mark it as a _ordered factor_ to specify that 'Up' is positive and 'Down' negative (for the confusion matrix later):

y = coerce(y, OrderedFactor)
classes(y[1])

# Note that in this case the default order comes from the lexicographic order which happens  to map  to  our intuition since `D`  comes before `U`.

cm = countmap(y)
categories, vals = collect(keys(cm)), collect(values(cm))
Plots.bar(categories, vals, title="Bar Chart Example", legend=false)
ylabel!("Number of occurrences")

# \fig{ISL-lab-4-bal.svg}
#
# Seems pretty balanced.

# Let's now try fitting a simple logistic classifier (aka logistic regression) not using `:Year` and `:Today`:

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
X2 = select(X, Not([:Year, :Today]))
classif = machine(LogisticClassifier(), X2, y)

# Let's fit it to the data and try to reproduce the output:

fit!(classif)
ŷ = MLJ.predict(classif, X2)
ŷ[1:3]

# Note that here the `ŷ` are _scores_.
# We can recover the average cross-entropy loss:

cross_entropy(ŷ, y) |> mean |> r3

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
@show accuracy(ŷ, y)  |> r3
@show accuracy(cm)    |> r3  # same thing
@show positive_predictive_value(ŷ, y) |> r3   # a.k.a. precision
@show recall(ŷ, y)    |> r3
@show f1score(ŷ, y)   |> r3

# Let's now train on the data before 2005 and use it to predict on the rest.
# Let's find the row indices for which the condition holds

train = 1:findlast(X.Year .< 2005)
test = last(train)+1:length(y);

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

Xnew = (Lag1 = [1.2, 1.5], Lag2 = [1.1, -0.8])
ŷ = MLJ.predict(classif, Xnew)
ŷ |> pprint

# **Note**: when specifying data, we used a simple `NamedTuple`; we could also have defined a dataframe or any other compatible tabular container.
# Note also that we retrieved the raw predictions here i.e.: a score for each class; we could have used `predict_mode` or indeed

mode.(ŷ)



# ### LDA


#
# Let's do a similar thing but with a LDA model this time:

BayesianLDA = @load BayesianLDA pkg=MultivariateStats

classif = machine(BayesianLDA(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3

# Note: `BayesianLDA` is LDA using a multivariate normal model for each class with a default prior inferred from the proportions for each class in the training data.
# You can also use the bare `LDA` model which does not make these assumptions and allows using a different metric in the transformed space, see the docs for details.

LDA = @load LDA pkg=MultivariateStats
using Distances

classif = machine(LDA(dist=CosineDist()), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3



# ### QDA


#
# Bayesian QDA is available via ScikitLearn:

BayesianQDA = @load BayesianQDA pkg=MLJScikitLearnInterface

# Using it is done in much the same way as before:

classif = machine(BayesianQDA(), X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3



# ### KNN


#
# We can use K-Nearest Neighbors models via the [`NearestNeighbors`](https://github.com/KristofferC/NearestNeighbors.jl) package:

KNNClassifier = @load KNNClassifier

knnc = KNNClassifier(K=1)
classif = machine(knnc, X3, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# Pretty bad... let's try with three neighbors

knnc.K = 3
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)
accuracy(ŷ, y[test]) |> r3

# A bit better but not hugely so.





# ## Caravan insurance data


#
# The caravan dataset is part of ISLR as well:

caravan  = dataset("ISLR", "Caravan")
size(caravan)

# The target variable is `Purchase`, effectively  a categorical

purchase = caravan.Purchase
vals     = unique(purchase)

# Let's see how many of each we have

nl1 = sum(purchase .== vals[1])
nl2 = sum(purchase .== vals[2])
println("#$(vals[1]) ", nl1)
println("#$(vals[2]) ", nl2)

# we can also visualise this as was done before:

cm = countmap(purchase)
categories, vals = collect(keys(cm)), collect(values(cm))
bar(categories, vals, title="Bar Chart Example", legend=false)
ylabel!("Number of occurrences")

# \fig{ISL-lab-4-bal2.svg}

# that's quite unbalanced.
#
# Apart from the target, all other variables are numbers; we can standardize the data:

y, X = unpack(caravan, ==(:Purchase))

mstd = machine(Standardizer(), X)
fit!(mstd)
Xs = MLJ.transform(mstd, X)

var(Xs[:,1]) |> r3

# **Note**: in MLJ, it is recommended to work with pipelines / networks when possible and not do "step-by-step" transformation and fitting of the data as this is more error prone. We do it here to stick to the ISL tutorial.
#
# We split the data in the first 1000 rows for testing and the rest for training:

test = 1:1000
train = last(test)+1:nrows(Xs);

# Let's now fit a KNN model and check the misclassification rate

classif = machine(KNNClassifier(K=3), Xs, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3

# that looks good but recall the problem is very unbalanced

mean(y[test] .!= "No") |> r3

# Let's fit a logistic classifier to this problem

classif = machine(LogisticClassifier(), Xs, y)
fit!(classif, rows=train)
ŷ = predict_mode(classif, rows=test)

accuracy(ŷ, y[test]) |> r3


# ### ROC and AUC


#
# Since we have a probabilistic classifier, we can also check metrics that take _scores_ into account such as the area under the ROC curve (AUC):

ŷ = MLJ.predict(classif, rows=test)

auc(ŷ, y[test])

# We can also display the curve itself

fprs, tprs, thresholds = roc_curve(ŷ, y[test])

plot(fprs, tprs, linewidth=2, size=(800,600))
xlabel!("False Positive Rate")
ylabel!("True Positive Rate")






