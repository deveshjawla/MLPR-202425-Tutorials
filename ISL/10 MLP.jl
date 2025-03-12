# Classification of MNIST dataset using an MLP and Lux.jl

using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots
using CSV, DataFrames
rng = Xoshiro(1)

function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end
function mnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x4dim = meanpool((x4dim), (2, 2))
    x4dim = flatten(x4dim)
    # ys = permutedims(data.label) .+ 1

    yhot = onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix
    return DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
    # return x4dim, ys
end
train = CSV.read("Data/Tabular/mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("Data/Tabular/mnist/mnist_test.csv", DataFrame, header=1)

#===== MODEL =====#

model = Chain(
    Dense(196 => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 10),
)

#===== METRICS =====#

const lossfn = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y)
        predicted_class = onecold(Array(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

#===== TRAINING =====#

train_dataloader, test_dataloader = mnistloader(train, 512), mnistloader(test, 10000)
ps, st = Lux.setup(rng, model)

vjp = AutoZygote()

train_state = Training.TrainState(model, ps, st, OptimiserChain(WeightDecay(3e-4), AdaBelief()))

model_compiled = model

### Lets train the model
nepochs = 40
tr_acc, te_acc = 0.0, 0.0
for epoch in 1:nepochs
    stime = time()
    for (x, y) in train_dataloader
        _, _, _, train_state = Training.single_train_step!(
            vjp, lossfn, (x, y), train_state
        )
    end
    ttime = time() - stime

    tr_acc = accuracy(
        model_compiled, train_state.parameters, train_state.states, train_dataloader) *
             100
    te_acc = accuracy(
        model_compiled, train_state.parameters, train_state.states, test_dataloader) *
             100

    trained_parameters, trained_states = deepcopy(train_state.parameters), deepcopy(train_state.states)
    if epoch % 5 == 0
        @save "Data/Tabular/mnist/MLP_trained_model_$(epoch).jld2" trained_parameters trained_states
        println("saved to ", "trained_model_$(epoch).jld2")
    end

    @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: \
             %.2f%%\n" epoch nepochs ttime tr_acc te_acc
end
