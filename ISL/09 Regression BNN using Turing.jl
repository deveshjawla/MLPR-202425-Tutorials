using Flux # Deep Learning Library, Lux.jl is another among others
using StatsPlots
using Distributions
using Turing # For Bayesian Inference

### Create Experimental Data

begin
	f(x) = cos(x) + rand(Normal(0, 0.1)) #y
	xTrain = collect(-3:0.1:3)
	yTrain = f.(xTrain)
	plot(xTrain, yTrain, seriestype = :scatter, label = "Train Data", xlim = [-10, 10])
	plot!(xTrain, cos.(xTrain), label = "Truth")
end

nn = Chain(Dense(1 => 2, sigmoid), Dense(2 => 1, bias = false)) # Neural Network 1 -> 2 -> 1

init_params, re = Flux.destructure(nn)

const n_params = lastindex(init_params)

@model function bayes_nn(xs, ys)
	nn_params ~ MvNormal(zeros(n_params), ones(n_params)) #Prior
	nn = re(nn_params) #Build the net
	preds = nn(xs) #Predictions
	sigma ~ Gamma(0.1, 1 / 0.1) # Prior for the variance
	for i ∈ 1:lastindex(ys)
		ys[i] ~ Normal(preds[i], sigma)
	end
end

using ReverseDiff # For Automatic Differentiation, Mooncake, Zygote, ReverseDiff, ForwardDiff, etc. are the AD libraries in Julia. This is ordered by speed, Mooncake is the fastest.

N = 1000 # Number of MCMC samples to generate, the more the better.
model = bayes_nn(permutedims(xTrain), yTrain)
chain = sample(model, NUTS(; adtype = AutoReverseDiff()), N)

lp, maxInd = findmax(chain[:lp]) # Find the index of the maximum log posterior (log likelihood using SGD) (MAP Estimate) Maximum A Posteriori

params, internals = chain.name_map
bestParams = map(x -> chain[x].data[maxInd], params[1:6])
nn = re(bestParams) # Build the neural network with the MAP estimate, thats also what you would have got using Frequnetist Deep Learning techniques such as SGD
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

#HW use parallel computing to make BNN Inference faster