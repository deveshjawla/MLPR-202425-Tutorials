# ## Getting started

using MLJ
import RDatasets: dataset
import DataFrames: DataFrame, select, Not, describe
using Random
using StatsPlots

data = dataset("datasets", "USArrests")
names(data) |> println

# Let's have a look at the mean and standard deviation of each feature:

describe(data, :mean, :std) |> show

# Let's extract the numerical component and coerce

X = select(data, Not(:State))
X = coerce(X, :UrbanPop => Continuous, :Assault => Continuous)

# ## PCA pipeline
#
# PCA is usually best done after standardization but we won't do it here:

PCA = @load PCA pkg = MultivariateStats

pca_mdl = Standardizer |> PCA(variance_ratio = 1)
pca = machine(pca_mdl, X)
fit!(pca)
PCA
W = MLJ.transform(pca, X)

# W is the PCA'd data; here we've used default settings for PCA:

schema(W).names

# Let's inspect the fit:

r = report(pca)
cumsum(r.pca.principalvars ./ r.pca.tvar)

# In the second line we look at the explained variance with 1 then 2 PCA features and it seems that with 2 we almost completely recover all of the variance.

# ## More interesting data...

# Instead of just playing with toy data, let's load the orange juice data and extract only the columns corresponding to price data:

data = dataset("ISLR", "OJ")

feature_names = [
	:PriceCH, :PriceMM, :DiscCH, :DiscMM, :SalePriceMM, :SalePriceCH,
	:PriceDiff, :PctDiscMM, :PctDiscCH,
]

X = select(data, feature_names)
y = select(data, :Purchase)

train, test = partition(eachindex(y.Purchase), 0.7, shuffle = true, rng = 1515)

using StatsBase
countmap(y.Purchase)
# ### PCA pipeline

Random.seed!(1515)

SPCA = Pipeline(
	Standardizer(),
	PCA(variance_ratio = 1.0),
)

spca = machine(SPCA, X)
fit!(spca)
W = MLJ.transform(spca, X)
names(W)

# What kind of variance can we explain?

rpca = report(spca).pca
cs = cumsum(rpca.principalvars ./ rpca.tvar)


# Let's visualise this

using Plots
begin
	Plots.bar(1:length(cs), cs, legend = false, size = ((800, 600)), ylim = (0, 1.1))
	xlabel!("Number of PCA features")
	ylabel!("Ratio of explained variance")
	plot!(1:length(cs), cs, color = "red", marker = "o", linewidth = 3)
end
# So 4 PCA features are enough to recover most of the variance.

### HW TODO - Test the performance using LogisticClassifier and compare the performance on PCA features and the original set of features

# ## Caravan insurance data

# The caravan dataset is part of ISLR as well:
caravan = dataset("ISLR", "Caravan")
size(caravan)

# The target variable is `Purchase`, effectively  a categorical

purchase = caravan.Purchase
vals = unique(purchase)

# Let's see how many of each we have

nl1 = sum(purchase .== vals[1])
nl2 = sum(purchase .== vals[2])
println("#$(vals[1]) ", nl1)
println("#$(vals[2]) ", nl2)

# we can also visualise this as was done before:

begin
	cm = countmap(purchase)
	categories, vals = collect(keys(cm)), collect(values(cm))
	bar(categories, vals, title = "Bar Chart Example", legend = false)
	ylabel!("Number of occurrences")
end
# that's quite unbalanced.
#
# Apart from the target, all other variables are numbers; we can standardize the data:

y, X = unpack(caravan, ==(:Purchase))

mstd = machine(Standardizer(), X)
fit!(mstd)
Xs = MLJ.transform(mstd, X)

var(Xs[:, 1])

density(Xs.MAANTHUI, legend = false)

# **Note**: in MLJ, it is recommended to work with pipelines / networks when possible and not do "step-by-step" transformation and fitting of the data as this is more error prone. We do it here to stick to the ISL tutorial.
#
# We split the data in the first 1000 rows for testing and the rest for training:

test = 1:1000
train = last(test)+1:nrows(Xs)

KNNClassifier = @load KNNClassifier

# Let's now fit a KNN model and check the misclassification rate

classif = machine(KNNClassifier(K = 3), Xs, y)
fit!(classif, rows = train)
ŷ = predict_mode(classif, rows = test)

accuracy(ŷ, y[test])

# that looks good but recall that the problem is very unbalanced
bacc(ŷ, y[test])

### Without Data leakage
train, test = partition(eachindex(y), 0.7, shuffle = true, rng = 1515)
mstd = machine(Standardizer(), X)
fit!(mstd, rows = train)
Xstrain = MLJ.transform(mstd, X[train, :])
Xstest = MLJ.transform(mstd, X[test, :])



classif = machine(KNNClassifier(K = 3), Xstrain, y[train])
fit!(classif)
ŷ = predict_mode(classif, Xstest)
accuracy(ŷ, y[test])
bacc(ŷ, y[test])

# Let's fit a logistic classifier to this problem

LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels

classif = machine(LogisticClassifier(), Xs, y)
fit!(classif, rows = train)
ŷ = predict_mode(classif, rows = test)

accuracy(ŷ, y[test])
bacc(ŷ, y[test])

### Without Data leakage
train, test = partition(eachindex(y), 0.7, shuffle = true, rng = 1515)
mstd = machine(Standardizer(), X)
fit!(mstd, rows = train)
Xstrain = MLJ.transform(mstd, X[train, :])
Xstest = MLJ.transform(mstd, X[test, :])

classif = machine(LogisticClassifier(), Xstrain, y[train])
fit!(classif)
ŷ = predict_mode(classif, Xstest)
accuracy(ŷ, y[test])
bacc(ŷ, y[test])

# ### ROC and AUC

# Since we have a probabilistic classifier, we can also check metrics that take _scores_ into account such as the area under the ROC curve (AUC):

ŷ = MLJ.predict(classif, Xstest)
ŷ = MLJ.predict_mode(classif, Xstest)

auc(ŷ, y[test])

# We can also display the curve itself

fprs, tprs, thresholds = roc_curve(ŷ, y[test])

begin
	plot(fprs, tprs, linewidth = 2, size = (600, 600))
	xlabel!("False Positive Rate")
	ylabel!("True Positive Rate")
end
