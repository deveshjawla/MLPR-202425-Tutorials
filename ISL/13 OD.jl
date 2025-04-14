using MLJ
using OutlierDetection
using OutlierDetectionNeighbors
using OutlierDetectionData
using StatisticalMeasures: area_under_curve
using StatsPlots

using OutlierDetectionData: ODDS
# ODDS.download.(ODDS.list(), force_accept = true)
X, y = ODDS.read("annthyroid")
y = replace(y, -1 => "outlier")
y = replace(y, 1 => "normal")
y = coerce(y, OrderedFactor{2})

# use 50% of the data for training
train, test = partition(eachindex(y), 0.5, shuffle = true)

#Define a Model such as a KNN
KNN = @iload KNNDetector pkg = OutlierDetectionNeighbors #n_components = number of classes or number of modes of the multivariate gaussian
knn = KNN()
mach = machine(knn, X, y)
fit!(mach, rows = train)
scores = transform(mach, rows = test)
scores_train, scores_test = scores
last(scores |> scale_minmax)
last(scores |> classify_quantile(0.9))

knn = ProbabilisticDetector(knn1 = KNN(k = 5), knn2 = KNN(k = 10),
	normalize = scale_minmax,
	combine = combine_mean)
mach = machine(knn, X[:, train], y[train])
fit!(mach)
predict(mach, X[:, test])

#TunedModel from MLJ does not work. Due to some out of date dependencies, or some bug in the code.