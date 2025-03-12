using Flux
using StatsPlots
using Distributions
using Turing

### Create Experimental Data

begin
	f(x) = cos(x) + rand(Normal(0, 0.1))
	xTrain = collect(-3:0.1:3)
	yTrain = f.(xTrain)
	plot(xTrain, yTrain, seriestype = :scatter, label = "Train Data")
	plot!(xTrain, cos.(xTrain), label = "Truth")
end

nn = Chain(Dense(1 => 2, tanh), Dense(2 => 1, bias = false))

init_params, re = Flux.destructure(nn)

n_params = lastindex(init_params)

@model function bayes_nn(xs, ys, n_params)
	nn_params ~ MvNormal(zeros(n_params), ones(n_params)) #Prior
	nn = re(nn_params)
	preds = nn(xs) #Build the net
	sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
	for i ∈ 1:lastindex(ys)
		ys[i] ~ Normal(preds[i], sigma)
	end
end

using ReverseDiff
N = 500
chain = sample(bayes_nn(permutedims(xTrain), yTrain, n_params), NUTS(; adtype = AutoReverseDiff()), N)

lp, maxInd = findmax(chain[:lp])

params, internals = chain.name_map
bestParams = map(x -> chain[x].data[maxInd], params[1:6])
nn = re(bestParams)
ŷ = nn(permutedims(xTrain))

xPlot = sort(xTrain)

begin
	sp = plot()
	for i in max(1, (maxInd[1] - 100)):min(N, (maxInd[1] + 100))
		paramSample = map(x -> chain[x].data[i], params[1:6])
		nn = re(paramSample)
		plot!(sp, xPlot, Array(nn(permutedims(xPlot))'), label = :none, colour = "lightblue")
	end

	plot!(xTrain, cos.(xTrain), seriestype = :line, label = "True", colour = "green", linewidth = 2)
	plot!(xTrain, permutedims(ŷ), seriestype = :line, label = "MAP Estimate", colour = "orange", linewidth = 2)
	plot!(sp, xTrain, yTrain, seriestype = :scatter, label = "Training Data", colour = "yellow")
end
lPlot = plot(chain[:lp], label = "Chain", title = "Log Posterior")
sigPlot = plot(chain[:sigma], label = "Chain", title = "Variance")
