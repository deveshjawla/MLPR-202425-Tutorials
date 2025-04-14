# Classification of MNIST dataset using a convolutional neural network,
# which is a variant of the original LeNet from 1998.

using Flux, JLD2, StatsPlots
using CSV, DataFrames, Statistics

folder = "mnist"  # sub-directory in which to save
isdir(folder) || mkdir(folder)
filename = joinpath(folder, "lenet.jld2")

#===== DATA =====#

# Calling MLDatasets.MNIST() will dowload the dataset if necessary,
# and return a struct containing it.
# It takes a few seconds to read from disk each time, so we do this once:

train_data = CSV.read(joinpath(folder, "mnist_train.csv"), DataFrame)  # i.e. split=:train
test_data = CSV.read(joinpath(folder, "mnist_test.csv"), DataFrame)

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# Flux needs a 4D array, with the 3rd dim for channels -- here trivial, grayscale.
# Combine the reshape needed with other pre-processing:

function loader(data::DataFrame; batchsize::Int = 512)
	x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim

	x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims = 1), x4dim, dims = (1, 2))

	yhot = Flux.onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix
	Flux.DataLoader((x4dim, yhot); batchsize, shuffle = true)
end

loader(train_data)  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(loader(test_data)); # (28×28×1×64 Array{Float32, 3}, 10×64 OneHotMatrix(::Vector{UInt32}))
train_data_loader = loader(train_data)


#===== MODEL =====#

# LeNet has two convolutional layers, and our modern version has relu nonlinearities.
# After each conv layer there's a pooling step. Finally, there are some fully connected (Dense) layers:

lenet = Chain(
	Conv((5, 5), 1 => 6, relu), #all filter sizes, number of layers are all also Hyperparameters
	MaxPool((2, 2)),
	Conv((5, 5), 6 => 16, relu),
	MaxPool((2, 2)),
	Flux.flatten,
	Dense(256 => 120, relu), #256 here is decided by the original image size, and then the convolutional part, how many layers, what filter sizes, Poolings etc.
	Dense(120 => 84, relu),
	Dense(84 => 10),
)

#===== ARRAY SIZES =====#

# A layer like Conv((5, 5), 1=>6) takes 5x5 patches of an image, and matches them to each
# of 6 different 5x5 filters, placed at every possible position. These filters are here:

Conv((5, 5), 1 => 6).weight # 5×5×1×6 Array{Float32, 4}

# map(w->w[:,:,1]*x1[11:15,11:15,1,1], eachslice(Conv((5, 5), 1 => 6).weight, dims=4)) # Explanation - We multiply a convolutional filters weight matrix with a cross section of an image, and it returns the output for each channel. In this example we have 6 such channels.

# This layer can accept any size of image; let's trace the sizes with the actual input:

#=

julia> x1 |> size
(28, 28, 1, 64)

julia> lenet[1](x1) |> size  # after Conv((5, 5), 1=>6, relu),
(24, 24, 6, 64)

julia> lenet[1:2](x1) |> size  # after MaxPool((2, 2))
(12, 12, 6, 64)

julia> lenet[1:3](x1) |> size  # after Conv((5, 5), 6 => 16, relu)
(8, 8, 16, 64)

julia> lenet[1:4](x1) |> size  # after MaxPool((2, 2))
(4, 4, 16, 64)

julia> lenet[1:5](x1) |> size  # after Flux.flatten 
(256, 64)

=#

# Flux.flatten is just reshape, preserving the batch dimesion (64) while combining others (4*4*16).
# This 256 must match the Dense(256 => 120). Here is how to automate this, with Flux.outputsize:

#===== METRICS =====#

# We're going to log accuracy and loss during training. There's no advantage to
# calculating these on minibatches, since MNIST is small enough to do it at once.

using Statistics: mean  # standard library

function loss_and_accuracy(model, data)
	(x, y) = only(loader(data; batchsize = size(data, 1)))  # make one big batch
	ŷ = model(x)
	loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
	#Flux=>Lux: logitcrossentropy => CrossEnrtropy(logits=true); crossentropy=>CrossEnrtropy(logits=false)
	acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits = 2)
	return loss, acc
end

@show loss_and_accuracy(lenet, test_data)  # accuracy about 10%, before training

#===== TRAINING =====#

# Let's collect some hyper-parameters in a NamedTuple, just to write them in one place.
# Global variables are fine -- we won't access this from inside any fast loops.

settings = (;
	eta = 0.001,     # learning rate
	lambda = 3e-4,  # for weight decay
	batchsize = 512,
	epochs = 20,
)
train_log = []

# Initialise the storage needed for the optimiser:
opt_rule = AdamW(settings.eta, (0.9, 0.999), settings.lambda)
opt_state = Flux.setup(opt_rule, lenet)
for epoch in 1:settings.epochs
	# @time will show a much longer time for the first epoch, due to compilation
	@time for (x, y) in train_data_loader
		grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
		Flux.update!(opt_state, lenet, grads[1])
	end

	# Logging & saving, but not on every epoch
	if epoch % 2 == 1
		loss, acc = loss_and_accuracy(lenet, train_data)
		test_loss, test_acc = loss_and_accuracy(lenet, test_data)
		@info "logging:" epoch acc test_acc
		nt = (; epoch, loss, acc, test_loss, test_acc)  # make a NamedTuple
		push!(train_log, nt)
	end
	if epoch % 5 == 0
		JLD2.jldsave(filename; lenet_state = Flux.state(lenet))
		println("saved to ", filename, " after ", epoch, " epochs")
	end
end

#HW TODO - Check why the MLP using Lux.jl was faster than CNN using Flux?

@show train_log

# We can re-run the quick sanity-check of predictions:
y1hat = softmax(lenet(x1))
@show hcat(Flux.onecold(y1hat, 0:9), Flux.onecold(y1, 0:9))

#===== LOADING TRAINED MODEL (SKIP THIS IF WANT TO TRAIN ANEW)=====#

# During training, the code above saves the model state to disk. Load the last version:

loaded_state = JLD2.load(filename, "lenet_state")

# Now you would normally re-create the model, and copy all parameters into that.
# We can use lenet2 from just above:

Flux.loadmodel!(lenet, loaded_state)

#===== THE END =====#

# 1. Cross-validation when Hyperparameter tuning - Requires Train and validation sets, ususally comes from the original train set, which are split for n folds
# 2. Evaluation 10-fold (evaluate!) when presenting a model to someone - Requires only train set which is split into 10 folds