# They want you to Train on a train set, and they want to benchmark all algorithms on a validation set (ideally this is hidden from competitors, or made unavailable by the client because they want to validation, or it is still in the future).
# 1. Cross-validation when Hyperparameter tuning - Requires Train and validation sets, ususally comes from the original train set, which are split for n folds
# 2. Evaluation 10-fold (evaluate!) when presenting a model to someone - Requires only train set which is split into 10 folds

# Classification of MNIST dataset using an MLP and Lux.jl

using Distributed
@everywhere n_splits = 8
addprocs(n_splits)
@everywhere begin
	n_splits = $n_splits
	using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots
	using CSV, DataFrames
	using SharedArrays
	rng = Xoshiro(1)
	function flatten(x::AbstractArray)
		return reshape(x, :, size(x)[end])
	end

	function mnistloader(data::DataFrame, batch_size_, shuffle_)
		x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim
		x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims = 1), x4dim, dims = (1, 2))
		x4dim = meanpool((x4dim), (2, 2)) #this is being done from experience to reduce dimensionality, you can do it by trial and error also. but we make 75% efficiency just by doing this.
		x4dim = flatten(x4dim)
		# ys = permutedims(data.label) .+ 1

		yhot = onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix
		return DataLoader((x4dim, yhot); batchsize = batch_size_, shuffle = shuffle_)
		# return x4dim, ys
	end

	"""
	If we want to do n-fold evaluation using a single validation set
	"""
	function split_indices(n_samples::Int, n_folds::Int)
		# Ensure valid input
		@assert n_folds > 1 "Number of folds should be greater than 1"
		@assert n_samples >= n_folds "Number of samples should be greater or equal to number of folds"

		# Generate an array of indices
		indices = collect(1:n_samples)

		# Divide the indices into approximately equal folds
		folds = collect(Iterators.partition(indices, ceil(Int, n_samples / n_folds)))

		# Prepare the cross-validation splits
		cv_splits = []

		for i in 1:n_folds
			# Train set is the current fold
			train_set = folds[i]

			# Store the train/validation sets
			push!(cv_splits, train_set)
		end

		return cv_splits
	end

	"""
	Independent 10-fold cross validation / evaluation splits - May yield better results after Hyperparameter Tuning
	"""
	# Cross-validation function
	function cross_validation_indices(n_samples::Int, n_folds::Int)
		# Ensure valid input
		@assert n_folds > 1 "Number of folds should be greater than 1"
		@assert n_samples >= n_folds "Number of samples should be greater or equal to number of folds"

		# Generate an array of indices
		indices = collect(1:n_samples)

		# Divide the indices into approximately equal folds
		folds = collect(Iterators.partition(indices, ceil(Int, n_samples / n_folds)))

		# Prepare the cross-validation splits
		cv_splits = []

		for i in 1:n_folds
			# Train set is the current fold
			train_set = folds[i]

			# Test set is next fold
			val_set = folds[mod1(i + 1, n_folds)]

			# Store the train/validation sets
			push!(cv_splits, (train_set, val_set))
		end

		return cv_splits
	end

	"""
	Standard CV approach found in most literature
	"""
	function standard_cv_indices(df_size, n_folds = 10)
		fold_size = div(df_size, n_folds)
		# Prepare the cross-validation splits
		cv_splits = []
		for i in 1:n_folds
			test_indices = (fold_size*mod(i, n_folds))+1:fold_size*mod1((i + 1), n_folds)
			train_indices = collect(1:df_size)[Not(test_indices)]

			# Store the train/validation sets
			push!(cv_splits, (train_indices, test_indices))
		end
		return cv_splits
	end

	#===== METRICS =====#

	const lossfn = CrossEntropyLoss(; logits = Val(true))

	function accuracy(model, ps, st, dataloader)
		total_correct, total = 0, 0
		st = Lux.testmode(st)
		for (x, y) in dataloader
			target_class = onecold(y)
			predicted_class = onecold(softmax(Array(first(model(x, ps, st)))))
			total_correct += sum(target_class .== predicted_class)
			total += length(target_class)
		end
		return total_correct / total
	end

	function balanced_accuracy(y_true, y_pred) # not so robust function, but works in this example, may need some degugging for other problems
		if length(y_true) != length(y_pred)
			throw(ArgumentError("y_true and y_pred must have the same length"))
		end
		isempty(y_true) && throw(ArgumentError("y_true must not be empty"))

		classes = unique(y_true)
		n_classes = length(classes)
		sum_recall = 0.0

		for c in classes
			idx = (y_true .== c)
			total = sum(idx)
			tp = sum(idx .& (y_pred .== c))
			recall = tp / total
			sum_recall += recall
		end

		return sum_recall / n_classes
	end

	function bal_accuracy(model, ps, st, dataloader)
		baccs = []
		st = Lux.testmode(st)
		for (x, y) in dataloader
			target_class = onecold(y)
			predicted_class = onecold(softmax(Array(first(model(x, ps, st)))))
			append!(baccs, balanced_accuracy(target_class, predicted_class))
		end
		return mean(baccs)
	end

	train = CSV.read("./mnist/mnist_train.csv", DataFrame, header = 1)
	test = CSV.read("./mnist/mnist_test.csv", DataFrame, header = 1)
end
#===== MODEL =====#

model = Chain(
	Dense(196 => 128, relu),
	Dense(128 => 64, relu),
	Dense(64 => 10),
) #MLP

#===== TRAINING =====#

for approach in ["IndependentCV", "StandardCV"]

	println("Training $(approach) approach \n\n\n")

	mkpath("./mnist/Lux MLP trained models/Parallel across splits/$(approach)/")
	val_accuracies = SharedArray{Float64}(n_splits)
	bal_val_accuracies = SharedArray{Float64}(n_splits)
	@time @sync @distributed for split_no in 1:n_splits
		if approach == "IndependentCV"
			cv_indices = cross_validation_indices(size(train, 1), n_splits)
		elseif approach == "StandardCV"
			cv_indices = standard_cv_indices(size(train, 1), n_splits)
		end
		split_idx = cv_indices[split_no]

		# train_loader = mnistloader(train[split_idx, :], 512)
		# val_loader = mnistloader(validation, 10000)

		train_loader = mnistloader(train[split_idx[1], :], 512, true)
		val_loader = mnistloader(train[split_idx[2], :], 512, true)

		ps, st = Lux.setup(rng, model)
		vjp = AutoZygote()
		train_state = Training.TrainState(model, ps, st, OptimiserChain(WeightDecay(3e-4), AdaBelief()))

		### Lets train a model on each CPU
		nepochs = 50
		tr_acc, val_acc = 0.0, 0.0
		best_acc = 10.0
		best_bal_acc = 10.0
		last_improvement = 1 #seen at first epoch
		stime = time()

		for epoch in 1:nepochs
			for (x, y) in train_loader
				_, _, _, train_state = Training.single_train_step!(
					vjp, lossfn, (x, y), train_state,
				)
			end

			tr_acc = accuracy(model, train_state.parameters, train_state.states, train_loader) * 100
			val_acc = accuracy(model, train_state.parameters, train_state.states, val_loader) * 100

			bal_tr_acc = bal_accuracy(model, train_state.parameters, train_state.states, train_loader) * 100
			bal_val_acc = bal_accuracy(model, train_state.parameters, train_state.states, val_loader) * 100

			trained_parameters, trained_states = deepcopy(train_state.parameters), deepcopy(train_state.states)

			# New logic for training 
			# HW TODO early stopping
			if val_acc > best_acc
				@save "./mnist/Lux MLP trained models/Parallel across splits/$(approach)/MLP_trained_model_$(split_no).jld2" trained_parameters trained_states
				best_acc = val_acc
				best_bal_acc = bal_val_acc
				val_accuracies[split_no] = val_acc
				bal_val_accuracies[split_no] = bal_val_acc
				last_improvement = epoch
			end
		end
		ttime = time() - stime
		@printf "Split %1d took %.2fs to complete %2d epochs \t Best Accuracy: %.2f%% \t Best Balanced Accuracy: %.2f%% on the validation set was seen at epoch %2d\n" split_no ttime nepochs best_acc best_bal_acc last_improvement
	end

	begin
		@printf "Accuracy: %.2f%% ± %.2f%%\n" mean(val_accuracies) 3 * std(val_accuracies)
		@printf "Balanced Accuracy: %.2f%% ± %.2f%%\n" mean(bal_val_accuracies) 3 * std(bal_val_accuracies)
		println("For any new data I.I.D. data, 99% of the time our model's accuracy will be within the above intervals.")
	end

end

for approach in ["IndependentCV", "StandardCV"]
	println("Testing $(approach) approach")

	bal_accuracies = SharedArray{Float64}(n_splits)
	ensemble = SharedArray{Float64}(10, 10000, n_splits)
	@sync @distributed for i in 1:n_splits
		test_loader = mnistloader(test, 10000, false)
		X = first(mnistloader(test, 10000, false))[1]
		@load "./mnist/Lux MLP trained models/Parallel across splits/$(approach)/MLP_trained_model_$(i).jld2" trained_parameters trained_states
		bal_test_acc = bal_accuracy(model, trained_parameters, trained_states, test_loader) * 100
		preds = softmax(Array(first(model(X, trained_parameters, trained_states))))
		ensemble[:, :, i] = preds
		bal_accuracies[i] = bal_test_acc
	end
	@printf "Balanced Accuracy: %.2f%% ± %.2f%%\n" mean(bal_accuracies) 3 * std(bal_accuracies)
	y = first(mnistloader(test, 10000, false))[2]
	final_bacc = balanced_accuracy(onecold(mean(ensemble, dims = 3), 0:9), onecold(y, 0:9))
	@printf "Final Balanced Accuracy: %.2f%%\n" final_bacc * 100
	println(repeat("-", 100))
end

# HW TODO - Find ways to optimise this code so that it uses less memory. Create a pull request if your code demonstrates a substantial gain in efficiency of memeory usage, or time. Bonus task- Find a way to also benchmark the energy efficiency (kWh used) of the code and also test if your improvements work.