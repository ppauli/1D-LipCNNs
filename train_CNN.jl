"""
Main code to train vanilla and L2 regularized CNNs
"""

# Import packages
using Flux
using Flux:onehotbatch, OneHotMatrix, @epochs, throttle, crossentropy, params
using Statistics
using HDF5
using BSON: @save
using LinearAlgebra
using XLSX
using Dates

include("src/solve_SDP.jl")
include("src/utils.jl")

# Load Data
train_x, train_y, test_x, test_y = load_data()

# CNN architecture
channels = [1,2,3]
fc_layers = [60]
l = [3,3]

ρ_vec = [0.15 0.2 0.25]

for ii in eachindex(ρ_vec)
    ρ=ρ_vec[ii]
    for jj = 1:10
        str_pool = "average"
        # Set up model
        m = Chain(
            # First convolution, operating upon a 128x1 signal
            Conv((l[1], 1), channels[1]=>channels[2], pad=(2,0), relu),
            x -> meanpool(x[1:128,:,:,:], (2,1)),

            # Second convolution, operating upon a 64x1 signal
            Conv((l[2], 1), channels[2]=>channels[3], pad=(2,0), relu),
            x -> meanpool(x[1:64,:,:,:], (2,1)),

            # Reshape 3d tensor into a 2d one, at this point it should be (32, 1, 3, N)
            x -> permutedims(x, [3, 2, 1, 4]),
            x -> reshape(x, :, size(x, 4)),
            Dense(96, fc_layers[1],relu),
            Dense(fc_layers[1], 5),

            # Finally, softmax to get nice probabilities
            softmax,
        )

        # Define loss function, optimiser, and get params
        loss(x, y) = crossentropy(m(x), y) + ρ* norm(params(m))
        ps = Flux.params(m)

        # Comparison functions
        compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
        accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

        # To check progrress while training
        progress = () -> @show(loss(train_x, train_y), accuracy(test_x, test_y) ) # callback to show loss

        # Train model with three different leaning rates
        epos =[200, 200, 200]
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
    
        str2 = "CNN_L2_"*str_date*".bson"
        #str2 = "CNN_"*str_date*".bson" # use this name if ρ=0.0
        str_comb = str1 * str2

        @save str_comb m

        # Find Lipschitz upper bound
        status, γ_sol, obj_value = solve_SDP(m, 0; solver_name=:Mosek)

        L = sqrt(γ_sol)/2
        print("Lipschitz bound: "*string.(L))
        print("Accuracy: "*string.(acc))

        XLSX.openxlsx("trained_models\\results.xlsx", mode="rw") do xf
            sheet = xf[1]
            i = 1
            while ismissing(sheet[i,1]) == false
                i = i + 1
            end
            sheet[i,1] = [vec2str(channels), vec2str(fc_layers), vec2str(l), str_pool, vec2str(lrs), vec2str(epos), acc, L, ρ ,str2]
        end
    end
end