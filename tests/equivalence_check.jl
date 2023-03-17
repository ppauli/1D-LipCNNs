"""
In this file we perform a sanity check on Lip2weights
"""

using BSON: @load
using RecurrentEquilibriumNetworks
using Flux
using Flux: onehotbatch, OneHotMatrix
using HDF5
using LinearAlgebra
using Statistics

include("../src/1D_LipCNN.jl")
include("../src/1D_LipCNNmax.jl")
include("../src/utils.jl")

@load "trained_models/LipCNN_23_03_17_15_40.bson" m

# load data
train_x, train_y, test_x, test_y = load_data()

# Define a model with LCNN
nu = 1              # input channel size
nh_c = [2,3]        # channels convolutional layers
nh_f = [60]        # neurons fully connected layers

channels = [nu, nh_c...]
fc_layers = nh_f  

# Define loss function, optimiser, and get params
loss(x, y) = crossentropy(m(x), y) 

# Comparison functions
compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

acc = accuracy(test_x,test_y)

# Reformulation of LCNN as CNN
m2 = LipCNN2Weights(m,channels,fc_layers,1)
m3 = Chain(m2,softmax)

# Define loss function, optimiser, and get params
loss(x, y) = crossentropy(m3(x), y) 

# Comparison functions
compare2(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy2(x, y::OneHotMatrix) = mean(compare2(y, m3(x)))

acc2 =accuracy2(test_x,test_y)

error = acc-acc2