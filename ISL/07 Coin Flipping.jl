
begin
	using Distributions
	using Random
	Random.seed!(12) # Set seed for reproducibility
	using StatsPlots
end

### Make Synthetic Experimental data
p_true = 0.7 # The true probability of heads in a fair coin is 0.5.
N = 10
data = rand(Bernoulli(p_true), N) # Results of N coin flips, where 1 means heads, 0 means tails.

## Question - How would you find from the data if the coin is fair or not?

### Inference

using Turing

plot(Beta(1, 1), label = "Beta Dist")
plot!(Uniform(0, 1), label = "Uniform Dist", ylim = (0, 1))

# We will use this above inforamtion for Prior P(A) because we know that the p_ture can only range between 0 and 1
# We will use the data to update our prior to get the Posterior
# We will use the Posterior to make predictions

#P(B|A) - likelihood
# Unconditioned coinflip model with `N` observations.
@model function coinflip(; N::Int)
	# Our prior belief about the probability of heads in a coin toss.
	p ~ Uniform(0, 1)

	# Heads or tails of a coin are drawn from `N` independent and identically distributed Bernoulli distributions with success rate `p`.
	y ~ filldist(Bernoulli(p), N)
end

result_untrained = [rand(coinflip(; N)) for _ in 1:100]
@info "If the untrained coinflip experiment is run 100 times and then the p is avearged, then we get" mean([x.p for x in result_untrained])
# This is because we are using a P which is still the Prior, after training (this word does not really suit for Bayesian Inference but we use it to draw analogy with general ML) on the experimental data we will have the Posterior
#P(A|B) - Posterior

using ReverseDiff

coinflip(y::AbstractVector{<:Real}) = coinflip(; N = lastindex(y)) | (; y) # Define a helper function for the coinflip model. This is an example of multiple dispatch.
model = coinflip(data)
sampler = NUTS(; adtype = AutoReverseDiff(compile = true))
n_mcmc_samples = 1000 # number of Monte Carlo samples, you can increase this for more precise results

chain = Turing.sample(model, sampler, n_mcmc_samples, progress = false)
histogram(chain)

# Visualize a blue density plot of the approximate posterior distribution using HMC (see Chain 1 in the legend).
mean_p = round(mean(chain[:p]); sigdigits = 3)
std_p = round(2 * std(chain[:p]); sigdigits = 3)
density(chain[:p]; xlim = (0, 1), legend = :left, w = 2, c = :blue, label = "Posterior $(mean_p) ± $(std_p)")

# Visualize the true probability of heads in red.
vline!([p_true]; label = "True probability $(p_true)", c = :red)
