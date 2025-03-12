using MLJ
using OutlierDetection
using OutlierDetectionNeighbors
using OutlierDetectionData
using StatisticalMeasures: area_under_curve
using StatsPlots
PATH = @__DIR__
cd(PATH)

using OutlierDetectionData: ODDS
# ODDS.download.(ODDS.list(), force_accept = true)
X, y = ODDS.read("annthyroid")
y = coerce(y, Binary)
# use 50% of the data for training
train, test = partition(eachindex(y), 0.5, shuffle = true)

#Define a Model such as a KNN
oodd = OutlierDetectionNeighbors.KNNDetector() #n_components = number of classes or number of modes of the multivariate gaussian

#Make the model ProbabilisticDetector
proba_oodd = ProbabilisticDetector(oodd)

#Tuning the model
cv = StratifiedCV(nfolds = 5, shuffle = true, rng = 0)
r = range(proba_oodd, :(detector.k), values = [1, 2, 3, 4, 5:5:100...])
t = TunedModel(model = proba_oodd, resampling = cv, tuning = Grid(), range = r, acceleration = CPUThreads(), measure = area_under_curve)
m = machine(t, permutedims(X[:, train]), vec(y[train])) |> fit!

#Use the best trained model to predict a test dataset
report(m).best_history_entry
b = report(m).best_model
eval_report = MLJ.evaluate(b, X[test, :], vec(y[test, :]), resampling = cv, measure = area_under_curve)

# # OutlierDetection.jl provides helper functions to normalize the scores,
# # for example using min-max scaling based on the training scores
# oodd_probas = machine(proba_oodd, X[train, :], y[train, :]) |> fit!

# # predict outlier probabilities based on the test data
# # oodd_probs = MLJ.predict(oodd_probas, X)
# oodd_probs = OutlierDetection.transform(oodd_probas, X[test, :])[2] #probability of being an outlier

# # OutlierDetection.jl also provides helper functions to turn scores into classes,
# # for example by imposing a threshold based on the training data percentiles
# oodd_classifier = machine(DeterministicDetector(oodd), X) |> fit!
# # predict outlier classes based on the test data
# oodd_preds = MLJ.predict(oodd_classifier, X)