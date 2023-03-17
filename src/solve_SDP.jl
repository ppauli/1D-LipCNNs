using SCS, COSMO, MosekTools, JuMP, LinearAlgebra, Flux
using BSON: @load

function solve_SDP(m, maxpoolind; solver_name=:COSMO)

# Create variable
    if solver_name == :COSMO
        model = Model(COSMO.Optimizer)
    elseif solver_name == :Mosek
        model = Model(optimizer_with_attributes(Mosek.Optimizer))
    end

    L = Int(length(Flux.params(m))/2) # number of layers

    K = Any[]
    b = Any[]
    W = Any[]
    bW = Any[]
    A = Any[]
    B = Any[]
    C = Any[]
    D = Any[]
    Lc = 0

    for i = 1:L
        if length(size(Flux.params(m)[2*i-1])) == 4 # checks if it is a conv layer
            push!(K,Flux.params(m)[2*i-1])
            push!(b,Flux.params(m)[2*i])
            As,Bs,Cs,Ds = create_ss(K[i])
            push!(A,As)
            push!(B,Bs)
            push!(C,Cs)
            push!(D,Ds)
            Lc = i
        else
            push!(W,Flux.params(m)[2*i-1])
            push!(bW,Flux.params(m)[2*i])
        end
    end

    ϵ = 0.0000000001 
    @variable(model, γ>=0)

    Λ = Any[]
    P = Any[]
    Q = Any[]
    l = zeros(Int,L)
    ci = zeros(Int,Lc)
    co = zeros(Int,Lc)
    nx = zeros(Int,Lc)
    nu = zeros(Int,L)
    ny = zeros(Int,L)
    M = Any[]

    nu1 = size(K[1],3)    
    push!(Q, γ*Matrix(I, nu1, nu1))

    for i = 1:L
        if i<=Lc
            l[i],~,ci[i],co[i] = size(K[i])
            nx[i] = ci[i]*(l[i]-1)
            nu[i] = ci[i]
            ny[i] = co[i]
            Ps = @variable(model, [1:nx[i],1:nx[i]], PSD)
            push!(P,Ps)
        else
            ny[i],nu[i] = size(W[i-Lc])
        end
        if i < L
            λs = @variable(model, [1:ny[i]], lower_bound = 0.0)
            push!(Λ,Diagonal(λs))
            if maxpoolind == 1 && i<=Lc
                qs = @variable(model, [1:ny[i]], lower_bound = 0.0)
                Qs = Diagonal(qs)
            else
                Qs = @variable(model, [1:ny[i],1:ny[i]], PSD)
            end
            push!(Q,Qs)
            if i == Lc
                Qts = kron(Matrix(I,32,32),Qs)
                push!(Q,Qts)
            end
        end
        if i <= Lc
            push!(M, [P[i]-A[i]'*P[i]*A[i] -A[i]'*P[i]*B[i] -(Λ[i]*C[i])'; -B[i]'*P[i]*A[i] Q[i]-B[i]'*P[i]*B[i] -(Λ[i]*D[i])'; -Λ[i]*C[i] -Λ[i]*D[i] 2*Λ[i]-Q[i+1]])
            @constraint(model, M[i]-ϵ*Matrix(I, nx[i]+nu[i]+ny[i], nx[i]+nu[i]+ny[i]) ∈ PSDCone())
        elseif Lc < i < L
            push!(M,[Q[i+1] -(Λ[i]*W[i-Lc])'; -Λ[i]*W[i-Lc] 2*Λ[i]-Q[i+2]])
            @constraint(model, M[i]-ϵ*Matrix(I, ny[i]+nu[i], ny[i]+nu[i]) ∈ PSDCone())
        elseif i == L
            #push!(M,[Q[i+1] -W[i-Lc]'; -W[i-Lc] Matrix(I, ny[i], ny[i])])
            push!(M,Q[i+1]-W[i-Lc]'*W[i-Lc])
            @constraint(model, M[i]-ϵ*Matrix(I, nu[i], nu[i]) ∈ PSDCone())
        end
    end

    @objective(model, Min, γ);
    optimize!(model)

    status = JuMP.termination_status(model)
    γ_sol = JuMP.value.(γ)
    obj_value = JuMP.objective_value(model)

    eig_min = 1
    for i = 1:L
        eig_min = minimum([eig_min, minimum(eigvals(JuMP.value.(M[i])))])
    end
    Q_val = Any[]
    for i = 1:L+1
        push!(Q_val,JuMP.value.(Q[i]))
    end

    return status, γ_sol, obj_value, eig_min, Q_val
end