"""
This code is a debugging code
"""

using BSON: @load
using RecurrentEquilibriumNetworks
using Flux
using Flux: onehotbatch
using HDF5
using LinearAlgebra
using BlockDiagonals

include("../src/1D_LipCNN.jl")
include("../src/1D_LipCNNmax.jl")
include("../estimate_Lip_CNN.jl")
include("../src/utils.jl")


nu = 1              # input channel size
nh_c = [2,3]        # channels convolutional layers
nh_f = Int[60]    

channels = [nu, nh_c...]
fc_layers = nh_f
N = 128

train_x, train_y, test_x, test_y = load_data()

@load "trained_models/LipCNN_23_02_24_08_37.bson" m

m2 = LipCNN2Weights(m,channels,fc_layers,0)

status, γ_sol, obj_value, eig_min, Q_val = solve_SDP(m2, 0; solver_name=:Mosek)
L = sqrt(γ_sol)/2

x1 = reshape(test_x[:,1,1,1],128,1,1,1)
x2 = x1 + 0.001*sign.(randn(128,1,1,1)).*rand(128,1,1,1)
y1 = m2(x1)
y2 = m2(x2)

Lip_vec = zeros(1000)
for ii = 1:1000
    x1 = reshape(test_x[:,1,1,ii],:,1,1,1)
    x2 = x1 + 0.001*sign.(randn(128,1,1,1)).*rand(128,1,1,1)
    y1 = m2(x1)
    y2 = m2(x2)
    Lip_vec[ii]=norm(reshape(y1-y2,5))/norm(reshape(x1-x2,128))
end


layer1 = Chain(
    Conv((3, 1), 1=>2, pad=(2,0),relu),
    x -> x[1:128,:,:,:]
    )

Flux.params(layer1)[1] .= Flux.params(m2)[1]
Flux.params(layer1)[2] .= Flux.params(m2)[2]

u1_1 = layer1(x1)
u1_2 = layer1(x2)

Qsum1 = 0
Term1_1 = 0
Term1_2 = 0
for ii = 1:N
    dx = reshape(x1[ii,:,:,:]-x2[ii,:,:,:],1)
    du = reshape(u1_1[ii,:,:,:]-u1_2[ii,:,:,:],2)
    Term1_1 += 0.25*dx'*Q_val[1]*dx
    Term1_2 += 0.25*du'*Q_val[2]*du
    Qsum1 += 0.25*dx'*Q_val[1]*dx - 0.25*du'*Q_val[2]*du
end

layer2 = Chain(
    x -> meanpool(x, (2,1))
    )

u2_1 = layer2(u1_1)
u2_2 = layer2(u1_2)

Qsum2 = 0
Term2_1 = 0
Term2_2 = 0
for ii = 1:Int(N/2)
    dx = reshape([u1_1[2*(ii-1)+1,:,:,:] u1_1[2*ii,:,:,:]]-[u1_2[2*(ii-1)+1,:,:,:] u1_2[2*ii,:,:,:]],4)
    du = reshape(u2_1[ii,:,:,:]-u2_2[ii,:,:,:],2)
    Term2_1 += 0.25*dx'*BlockDiagonals.BlockDiagonal([Q_val[2],Q_val[2]])*dx
    Term2_2 += 0.5*du'*Q_val[2]*du
    Qsum2 = Qsum2 + 0.25*dx'*BlockDiagonals.BlockDiagonal([Q_val[2],Q_val[2]])*dx - 0.5*du'*Q_val[2]*du
end

layer3 = Chain(
    Conv((3, 1), 2=>3, pad=(2,0),relu),
    x -> x[1:64,:,:,:]
    )

Flux.params(layer3)[1] .= Flux.params(m2)[3]
Flux.params(layer3)[2] .= Flux.params(m2)[4]

u3_1 = layer3(u2_1)
u3_2 = layer3(u2_2)

Qsum3 = 0
Term3_1 = 0
Term3_2 = 0
for ii = 1:Int(N/2)
    dx = reshape(u2_1[ii,:,:,:]-u2_2[ii,:,:,:],2)
    du = reshape(u3_1[ii,:,:,:]-u3_2[ii,:,:,:],3)
    Term3_1 += 0.5*dx'*Q_val[2]*dx
    Term3_2 += 0.5*du'*Q_val[3]*du
    Qsum3 = Qsum3 + 0.5*dx'*Q_val[2]*dx - 0.5*du'*Q_val[3]*du
end

layer4 = Chain(
    x -> meanpool(x, (2,1))
    )

u4_1 = layer4(u3_1)
u4_2 = layer4(u3_2)

Qsum4 = 0
Term4_1 = 0
Term4_2 = 0
for ii = 1:Int(N/4)
    dx = reshape([u3_1[2*(ii-1)+1,:,:,:] u3_1[2*ii,:,:,:]]-[u3_2[2*(ii-1)+1,:,:,:] u3_2[2*ii,:,:,:]],6)
    du = reshape(u4_1[ii,:,:,:]-u4_2[ii,:,:,:],3)
    Term4_1 += 0.5*dx'*BlockDiagonals.BlockDiagonal([Q_val[3],Q_val[3]])*dx
    Term4_2 += du'*Q_val[3]*du
    Qsum4 = Qsum4 + 0.5*dx'*BlockDiagonals.BlockDiagonal([Q_val[3],Q_val[3]])*dx - du'*Q_val[3]*du
end

layer5 = Chain(
    x -> permutedims(x, [3, 2, 1, 4]),
    x -> reshape(x, :, size(x, 4)),
    )

u5_1 = layer5(u4_1)
u5_2 = layer5(u4_2)

Qsum5 = 0
Term5_1 = 0
Term5_2 = 0
for ii = 1:Int(N/4)
    dx = reshape(u4_1[ii,:,:,:]-u4_2[ii,:,:,:],3)
    Term5_1 += dx'*Q_val[3]*dx
end
du = reshape(u5_1-u5_2,96)
Term4_2 += du'*Q_val[4]*du
Qsum5 = Qsum5 + Term5_1 - du'*Q_val[4]*du


layer6 = Chain(
    Dense(96, fc_layers[1],relu)
)

Flux.params(layer6)[1] .= Flux.params(m2)[5]
Flux.params(layer6)[2] .= Flux.params(m2)[6]

u6_1 = layer6(u5_1)
u6_2 = layer6(u5_2)

dx = reshape(u5_1-u5_2,96)
du = reshape(u6_1-u6_2,60)
Qsum6 = dx'*Q_val[4]*dx-du'*Q_val[5]*du
Term6_1 = dx'*Q_val[4]*dx
Term6_2 = du'*Q_val[5]*du

layer7 = Chain(
    Dense(fc_layers[1], 5)
)

Flux.params(layer7)[1] .= Flux.params(m2)[7]
Flux.params(layer7)[2] .= Flux.params(m2)[8]

u7_1 = layer7(u6_1)
u7_2 = layer7(u6_2)

dx = reshape(u6_1-u6_2,60)
du = reshape(u7_1-u7_2,5)
Term7_1 = dx'*Q_val[5]*dx
Term7_2 = du'*du
Qsum7 = dx'*Q_val[5]*dx-du'*du

Lip_bed = L^2*reshape(x1-x2,128)'*reshape(x1-x2,128)-reshape(m2(x1)-m2(x2),5)'*reshape(m2(x1)-m2(x2),5)

Qsum = Qsum1 + Qsum2 + Qsum3 + Qsum4 + Qsum6 + Qsum7