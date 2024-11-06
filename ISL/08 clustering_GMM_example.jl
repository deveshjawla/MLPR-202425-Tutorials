begin
    using Distributions
    using FillArrays
    using StatsPlots
    using LinearAlgebra
    using Random
end
# Set a random seed.
Random.seed!(3)

# Define Gaussian mixture model.
w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

# We draw the data points.
N = 60
x = rand(mixturemodel, N)

scatter(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")

using Turing

@model function gaussian_mixture_model(x, num_clusters)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = num_clusters
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

using ReverseDiff
setadbackend(:reversediff)

model = gaussian_mixture_model(x, 2)

sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
nsamples = 100
nchains = 1
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains)

plot(chains[["μ[1]", "μ[2]"]]; colordim=:parameter, legend=true)
plot(chains[["w[1]", "w[2]"]]; colordim=:parameter, legend=true)

# Model with mean of samples as parameters.
μ_mean = [mean(chains, "μ[$i]") for i in 1:2]
w_mean = [mean(chains, "w[$i]") for i in 1:2]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ_mean], w_mean)

contour(
    range(-7.5, 3; length=1_000),
    range(-6.5, 3; length=1_000),
    (x, y) -> logpdf(mixturemodel_mean, [x, y]);
    widen=false
)
scatter!(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")

assignments = [mean(chains, "k[$i]") for i in 1:N]
scatter(
    x[1, :],
    x[2, :];
    legend=false,
    title="Assignments on Synthetic Dataset",
    zcolor=assignments
)

function pairs_to_matrix(X1, X2)
    n_pairs = lastindex(X1) * lastindex(X2)
    test_x_area = Matrix{Float32}(undef, 2, n_pairs)
    count = 1
    for x1 in X1
        for x2 in X2
            test_x_area[:, count] = [x1, x2]
            count += 1
        end
    end
    return test_x_area
end

ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, output_activation_function=output_activation_function)

test_x_area = pairs_to_matrix(x[1, :], x[2, :])
ŷ_uncertainty = edlc_uncertainty(m(test_x_area))
uncertainties = reshape(ŷ_uncertainty, (lastindex(X1), lastindex(X2)))
gr(size=(700, 600), dpi=300)
heatmap(X1, X2, uncertainties)
scatter!(train_x[1, train_y[1, :].==1], train_x[2, train_y[1, :].==1], color=:red, label="1")
scatter!(train_x[1, train_y[2, :].==1], train_x[2, train_y[2, :].==1], color=:green, label="2")
scatter!(train_x[1, train_y[3, :].==1], train_x[2, train_y[3, :].==1], color=:blue, label="3", legend_title="Classes", aspect_ratio=:equal, xlim=[-4, 4], ylim=[-3, 5], colorbar_title="Evidential Deep Learning Uncertainty")