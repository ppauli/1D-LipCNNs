"""
In this file we compare the state space representation to the Conv function
"""

using NNlib
using LinearAlgebra
using BSON: @load
using RecurrentEquilibriumNetworks
using LinearAlgebra
using Flux

include("../src/utils.jl")
include("../src/1D_LipCNN.jl")

# load model
@load "trained_models/LipCNN_23_03_07_19_36.bson" m

# define NN architecture
N = 128
cin = 1
cout = 2
kernel = 3
ny = cout
nu = cin
nx = (kernel-1)*cin

channels = [1,2,3]
fc_layers = [60]
maxpoolind = 0

# generate random input
u = randn(N,cin,1) # N, channels, 1

# load parameters of the first convolutional layer
Y = m[1].Ys[1]
Z = m[1].Zs[1]
H = m[1].Hs[1]
b = m[1].bs[1]
γ = m[1].ds[1]
l = m[1].l[1]
ρ = m[1].γ
Lm = 2*ρ*Matrix(I,nu,nu) # Factor 2 comes from 2 average pooling layers

# Comparison of different codes
K,b,L = LipCNN2CNN(Y,Z,H,γ,b,l,Lm)

x = zeros(N+1,nx,1)
y = zeros(N,ny,1)
A,B,C,D = create_ss(K)
for k = 1:N
    x[k+1,:,:] = A*x[k,:,:] + B*u[k,:,:]
    y[k,:,:] = C*x[k,:,:] + D*u[k,:,:] + b
end

m2 = Chain(
    Conv((3, 1), 1=>2, pad=(2,0)),
    )

Flux.params(m2)[1] .= K
Flux.params(m2)[2] .= b

y2t = m2(reshape(u,128,1,1,1))
y2 = reshape(y2t[1:128,:,:,:],128,2,1)

error = y-y2