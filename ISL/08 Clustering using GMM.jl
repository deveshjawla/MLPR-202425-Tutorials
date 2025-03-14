using Distributed
addprocs(8) #Add 8 processors
using Distributions
@everywhere using FillArrays # @everywhere because our Bayesian model uses FillArrays
using StatsPlots
using LinearAlgebra
using Random
@everywhere using Turing # For Bayesian Inference on all cores that's why @everywhere
@everywhere using ReverseDiff # For Automatic Differentiation on all cores that's why @everywhere

### Generate Synthetic Training Data
# Set a random seed.
Random.seed!(3)

# Define Gaussian mixture model in 2-d space (x,y) to get some synthentic data, and then we forget about it.
w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), [0.5, 1]) for μₖ in μ], w)# 2 dimensional Normal with mean along x,y = [-3.5, -3.5], and then the other blob is x,y = [0.5, 0.5]

begin
	using Plots

	# Define the grid for plotting
	x_range = range(-8, 5; length = 100)
	y_range = range(-9, 4; length = 100)
	z = [pdf(mixturemodel, [x, y]) for x in x_range, y in y_range]

	# Plot the 3D surface
	surface(x_range, y_range, z; title = "3D Density Plot of Mixture Model", xlabel = "X", ylabel = "Y", zlabel = "Density")
end

#MvNormal([-3.5, -3.5], ones(2))
#MvNormal([0.5, 0.5], ones(2))

# We draw the data points.
N = 60
x = rand(mixturemodel, N)
x .+= randn(size(x))

scatter(x[1, :], x[2, :]; legend = false, title = "Synthetic Dataset")
# @everywhere x = $x

@everywhere @model function GMM(x, K) #Gaussian Mixture Model which takes x and K as arguments
	μ ~ Bijectors.ordered(MvNormal(zeros(K), ones(K))) #Prior
	w ~ Dirichlet(K, 1.0) #Prior
	x ~ MixtureModel([MvNormal(Fill(μₖ, K), ones(K)) for μₖ in μ], w) #Posterior
end

K = 2
model = GMM(x, K)
sampler = NUTS(; adtype = AutoReverseDiff(compile = true))
nsamples = 1000
nchains = 8 #Repeat this example with 8 chains and show Distributed computing with Turing and Julia
ch = sample(model, sampler, MCMCDistributed(), nsamples, nchains)

# Model with mean of samples as parameters.
μ_mean = [mean(ch, "μ[$i]") for i in 1:K]
w_mean = [mean(ch, "w[$i]") for i in 1:K]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ_mean], w_mean)

using Flux: softmax
function sample_class(xi, dists, w)
	lvec = [(logpdf(d, xi) + log(w[i])) for (i, d) in enumerate(dists)]
	return argmax(softmax(lvec))
end

D, N = size(x)

dists = [MvNormal(Fill(μₖ, D), I) for μₖ in μ_mean]
weights = w_mean
assignments = [sample_class(x[:, i], dists, weights) for i in 1:N]

begin
	contour(
		range(-8, 5; length = 1_000),
		range(-9, 4; length = 1_000),
		(x, y) -> 100 + logpdf(mixturemodel_mean, [x, y]);
		levels = 20,
		fill = true,
		xlabel = "X",
		ylabel = "Y",
		title = "Density Plot of the fitted Mixture Model",
		clabels = true,
		size = (900, 800),
	)
	first_class = x[:, assignments.==1]
	second_class = x[:, assignments.==2]
	scatter!(first_class[1, :], first_class[2, :]; legend = true, label = "Class 1")
	scatter!(second_class[1, :], second_class[2, :]; legend = true, label = "Class 2")
end

#HW TODO for students - Play around and calculate the variances (scales or spreads) of the two blobs
