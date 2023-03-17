"""
This code is to calculate the lower bound on the Lipschitz constant empirically
"""

using BSON: @load
using RecurrentEquilibriumNetworks
using Flux
using Flux: onehotbatch
using HDF5
using LinearAlgebra

include("src/1D_LipCNN.jl")
include("src/1D_LipCNNmax.jl")
include("src/solve_SDP.jl")
include("src/utils.jl")

@load "trained_models/LipCNN_23_03_17_15_40.bson" m

# load data
train_x, train_y, test_x, test_y = load_data()

# Define a model with LCNN
nu = 1              # input channel size
nh_c = [2,3]        # channels convolutional layers
nh_f = [60]        # neurons fully connected layers

channels = [nu, nh_c...]
fc_layers = nh_f
maxpoolind = 1

# In case of a LipCNN we need the first line to define m2, for vanilla and L2 regularized NNs the second
m2 = LipCNN2Weights(m,channels,fc_layers,maxpoolind)
#m2 = LooseSoftmax(m,maxpoolind)

# Estimate Lipschitz bound 
#status, γ_sol, obj_value, eig_min, Q_val = solve_SDP(m2, 0; solver_name=:Mosek)
#L = sqrt(γ_sol)/2
#print(L)

Lip_vec = zeros(5000000)

for ii = 1:10000
    for jj = 1:500
        x1 = reshape(test_x[:,1,1,ii],:,1,1,1)
        x2 = x1 + 0.000001*sign.(randn(128,1,1,1)).*rand(128,1,1,1)
        y1 = m2(x1)
        y2 = m2(x2)
        Lip_vec[5*(ii-1)+jj]=norm(reshape(y1-y2,5))/norm(reshape(x1-x2,128))
    end
end

L_lower = maximum(Lip_vec)

