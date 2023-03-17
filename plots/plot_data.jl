using Flux: OneHotMatrix, onehotbatch, params, crossentropy
using BSON: @load
using HDF5
using Statistics
using Plots
using RecurrentEquilibriumNetworks
using Flux

include("..//src/attacks.jl")
include("..//src/utils.jl")


@load "trained_models//CNN_23_03_07_16_04.bson" m

# Load Data
train_x, train_y, test_x, test_y = load_data()

loss(x, y) = crossentropy(m(x), y)

# Comparison functions
compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

ind = 1
#x1 = reshape(test_x[:,1,ind],:,1,1)
x1 = reshape(test_x[:,1,1,ind],:,1,1,1)
y1 = reshape(test_y[:,ind],:,1,1)

x1_adv = FGSM(m, loss, x1, y1 ; ϵ = 0.005)

~,indtrue = findmax(y1)
~,indx = findmax(m(x1))
~,indxadv = findmax(m(x1_adv))


acc = accuracy(test_x,test_y)

plot(x1[:])
plot!(x1_adv[:])
title!("Ground Truth:"*string(indtrue[1])*", Prediction:"*string(indx[1])*", Adv:"*string(indxadv[1]))

