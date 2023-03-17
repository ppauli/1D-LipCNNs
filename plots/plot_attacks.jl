using Flux
using Flux: crossentropy, onehotbatch, params, OneHotMatrix
using BSON: @load
using HDF5
using Plots
using Statistics
using RecurrentEquilibriumNetworks
using CSV, Tables
using LinearAlgebra

include("../src/1D_LipCNN.jl")
include("../src/1D_LipCNNmax.jl")
include("../src/attacks.jl")
include("../src/utils.jl")

nu = 1

# Load Data
train_x, train_y, test_x, test_y = load_data()

#ϵ_vec = 300*[0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.013, 0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027, 0.029, 0.031, 0.033, 0.035, 0.037, 0.039, 0.041]
ϵ_vec = 300*[0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038, 0.042, 0.044, 0.046, 0.048, 0.05]

str_vec = ["CNN_23_03_07_16_30.bson","CNN_L2_23_03_08_12_27.bson","CNN_L2_23_03_08_11_35.bson","CNN_L2_23_03_08_13_15.bson","CNN_L2_23_03_08_12_42.bson"]
acc_adv = zeros(25,5)

for ii = 1:5
    print(ii/30)
    @load "trained_models//"*str_vec[ii] m
    acc_adv[:,ii] = attack_acc(m,ϵ_vec)
end

str_vec_L2_05 = ["CNN_L2_23_03_08_04_46.bson","CNN_L2_23_03_08_04_22.bson","CNN_L2_23_03_08_05_46.bson","CNN_L2_23_03_08_05_34.bson","CNN_L2_23_03_08_04_10.bson"]
acc_adv_L2_05 = zeros(25,5)

for ii = 1:5
    print((5+ii)/30)
    @load "trained_models//"*str_vec_L2_05[ii] m
    acc_adv_L2_05[:,ii] = attack_acc(m,ϵ_vec)
end

str_vec_L2_1 = ["CNN_L2_23_03_08_06_22.bson","CNN_L2_23_03_08_07_06.bson","CNN_L2_23_03_08_05_58.bson","CNN_L2_23_03_08_06_10.bson","CNN_L2_23_03_08_06_54.bson"]
acc_adv_L2_1 = zeros(25,5)

for ii = 1:5
    print((10+ii)/30)
    @load "trained_models//"*str_vec_L2_1[ii] m
    acc_adv_L2_1[:,ii] = attack_acc(m,ϵ_vec)
end

str_vec_Lip5 = ["LipCNN_23_03_07_09_27.bson","LipCNN_23_03_07_10_35.bson","LipCNN_23_03_07_09_49.bson","LipCNN_23_03_07_10_23.bson","LipCNN_23_03_07_10_46.bson"]
acc_adv_Lip5 = zeros(25,5)

for ii = 1:5
    print((15+ii)/30)
    @load "trained_models//"*str_vec_Lip5[ii] m
    acc_adv_Lip5[:,ii] = attack_acc(m,ϵ_vec)
end

str_vec_Lip10 = ["LipCNN_23_03_07_13_13.bson","LipCNN_23_03_07_12_51.bson","LipCNN_23_03_07_12_27.bson","LipCNN_23_03_07_13_37.bson","LipCNN_23_03_07_12_37.bson"]
acc_adv_Lip10 = zeros(25,5)

for ii = 1:5
    print((20+ii)/30)
    @load "trained_models//"*str_vec_Lip10[ii] m
    acc_adv_Lip10[:,ii] = attack_acc(m,ϵ_vec)
end

str_vec_Lip50 = ["LipCNN_23_03_07_21_48.bson","LipCNN_23_03_07_20_57.bson","LipCNN_23_03_07_21_28.bson","LipCNN_23_03_07_21_07.bson","LipCNN_23_03_07_21_38.bson"]
acc_adv_Lip50 = zeros(25,5)

for ii = 1:5
    print((25+ii)/30)
    @load "trained_models//"*str_vec_Lip50[ii] m
    acc_adv_Lip50[:,ii] = attack_acc(m,ϵ_vec)
end

plot(ϵ_vec, [acc_adv acc_adv_L2_1 acc_adv_Lip50], label=["nominal CNN" "L2-reg. CNN" "LipCNN"], linewidth=3)
#plot!(ϵ_vec_L2,acc_adv_L2)
#plot!(ϵ_vec_Lip,acc_adv_Lip)
xlabel!("perburbation strength ϵ")
ylabel!("accuracy")

savefig("PGD_plot.png") 

CSV.write("PGD_plot.csv", Tables.table([ϵ_vec'; acc_adv'; acc_adv_L2_05'; acc_adv_L2_1'; acc_adv_Lip5'; acc_adv_Lip10'; acc_adv_Lip50']))
