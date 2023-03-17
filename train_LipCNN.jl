"""
Main code to train a 1D-LipCNN
"""

# Import packages
using Flux
using Flux:onehotbatch, OneHotMatrix, throttle, crossentropy, params
using HDF5
using BSON: @save
using RecurrentEquilibriumNetworks
using LinearAlgebra
using Random
using Statistics
using XLSX
using Dates

include("src/1D_LipCNN.jl")
include("src/1D_LipCNNmax.jl")
include("src/utils.jl")
include("src/solve_SDP.jl")

# Load Data
train_x, train_y, test_x, test_y = load_data()

# Define a model with LCNN
nu = 1              # input channel size
nh_c = [2,3]        # channels convolutional layers
nh_f = Int[60]      # neurons fully connected layers
l = [3,3]           # kernel sze convolutional layers
N = size(train_x,1) # signal length            
ny = 5              # # of outputs
maxpoolind = 0

channels = [nu, nh_c...]
fc_layers = nh_f  

γ_vec = [5.0 10.0 50.0] # Lipschitz bound upper limit (must be a float)

for ii in eachindex(γ_vec)
    γ = γ_vec[ii]            
    for jj = 1:5
        if maxpoolind == 1
            LipCNN = LCNNmax{Float64}(nu, nh_c, nh_f, ny, l, N, γ)
            m = Chain(LipCNN, softmax)
            str_pool = "max"
        else
            LipCNN = LCNN{Float64}(nu, nh_c, nh_f, ny, l, N, γ)
            m = Chain(LipCNN, softmax)
            str_pool = "average"
        end

        # Define loss function, optimiser, and get params
        loss(x, y) = crossentropy(m(x), y) #+ 0.2* norm(params(m))
        ps = Flux.params(m)

        # Comparison functions
        compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
        accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

        # To check progress while training
        progress = () -> @show(loss(train_x, train_y), accuracy(test_x, test_y) ) # callback to show loss

        # Train model with three different leaning rates
        epos =[100, 200, 300]
        lrs = [1e-2, 1e-3, 1e-4]
        loss_on_test = []
        loss_on_train = []
        for j in eachindex(epos)
            opt = ADAM(lrs[j])
            for k = 1:epos[j]
                print("Epoch # "*string.(k))
                Flux.train!(loss, ps,[(train_x,train_y)], opt, cb = throttle(progress, 10))
                push!(loss_on_test, loss(test_x, test_y))
                push!(loss_on_train, loss(train_x, train_y))
            end
        end

        acc = accuracy(test_x,test_y)

        #Save model
        str_date = Dates.format(now(), "yy_mm_dd_HH_MM")
        str1 = "trained_models\\"
        if maxpool == 1
            str2 = "LipCNNmax_"*str_date*".bson"
        else
            str2 = "LipCNN_"*str_date*".bson"
        end
        str_comb = str1 * str2

        @save str_comb m loss_on_test loss_on_train

        # Reformulation of LipCNN as equivalent CNN
        m2 = LipCNN2Weights(m,channels,fc_layers,maxpoolind)

        # Find Lipschitz upper bound
        status, γ_sol, obj_value = solve_SDP(m2, maxpoolind; solver_name=:Mosek)

        if maxpool == 1
            L = sqrt(γ_sol)
        else
            L = sqrt(γ_sol)/2
        end
        print("Lipschitz bound: "*string.(L))
        print("Accuracy: "*string.(acc))

        XLSX.openxlsx("trained_models\\results.xlsx", mode="rw") do xf
            sheet = xf[1]
            i = 1
            while ismissing(sheet[i,1]) == false
                i = i + 1
            end
            sheet[i,1] = [vec2str(channels), vec2str(fc_layers), vec2str(l), str_pool, vec2str(lrs), vec2str(epos), acc, L,  γ ,str2]
        end
    end
end