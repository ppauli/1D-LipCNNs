"""
Direct parameters for an LCNN
"""
mutable struct LCNN{T} <: AbstractLBDN
    Ys::Tuple
    Zs::Tuple
    Hs::Tuple
    bs::Tuple
    ds::Tuple
    l::Vector{Int}
    γ::T
    nl::Union{typeof(relu), typeof(tanh)}
end

"""
    LCNN{T}(nu, nh, ny, γ; ...)

Constructor for an LBDN with nu inputs, nv outputs, and
`nh = [nh1, nh2,...]` specifying the size of hidden layers.
User-imposed Lipschitz bound `γ` has a default of 1.
"""
function LCNN{T}(
    nu::Int, 
    nh_c::Vector{Int},
    nh_f::Vector{Int},
    ny::Int,
    l::Vector{Int},
    N::Int,
    γ::T = T(1),
    init = Flux.glorot_uniform, 
    nl = Flux.relu,
    rng = Random.GLOBAL_RNG
) where T

    # Layer sizes
    lw_c = [nu, nh_c...]            # channel sizes all convolutional layers
    lw = [nu, nh_c..., nh_f..., ny]              # all layers
    lw_x = [nu, nh_c...].*[(l.-1)..., 0]   # state dimensions in covolutiinal layers
    L = length(lw)
    L_c = length(lw_c)

    # Initialise params
    Ys = fill(zeros(T,0,0), L-1)
    Zs = fill(zeros(T,0,0), L-1)
    Hs = fill(zeros(T,0,0), L_c-1)
    bs = fill(zeros(T,0), L-1)
    ds = fill(zeros(T,0), L-2)

    # Fill params
    for j in 1:L-1
        bs[j] = 1e-5*init(rng, lw[j+1] )
        (j<L-1) && (ds[j] = init(rng, lw[j+1] ))
        Zs[j] = init(rng, lw[j+1], lw[j+1] )
        if j < L_c
            Ys[j] = init(rng, lw[j]+lw_x[j], lw[j+1] )
            Hs[j] = init(rng, lw_x[j], lw_x[j] )
        elseif j == L_c
            Ys[j] = init(rng, Int(N/4)*lw[j], lw[j+1] )
        else
            Ys[j] = init(rng, lw[j], lw[j+1] )
        end
    end

    # Return params in tuple (immutable, nicer with Flux)
    return LCNN{T}(tuple(Ys...), tuple(Zs...), tuple(Hs...), tuple(bs...), tuple(ds...), l, γ, nl)
end

# Not exactly sure what this does...?
Flux.@functor LCNN

"""
    Flux.trainable(m::LCNN)

Define trainable parameters for an LCNN.
"""
Flux.trainable(m::LCNN) = (m.Ys, m.Zs, m.Hs, m.bs, m.ds)

"""
    (m::LCNN)(w)

Evaluate an LCNN given some input w.
"""
function (m::LCNN)(w::Union{T,AbstractVecOrMat{T}}) where T

    # Set up
    L = length(m.Ys)
    L_c = length(m.Hs)+1
    u = w
    nu = size(u,3)
    Ls = 2 * m.γ * Matrix(I, nu, nu) # Factor 2 comes from 2 meanpool layers
    r2 = sqrt(2)
    N = size(w,1)
    N2 = size(w,4)
    y = zeros(5,N2)

    # Loop the layers
    for i in 1:L # convolutional Layers
        if i < L_c
            Γ = Diagonal(m.ds[i]) # Why not just any diagonal matrix?
            U,V = Cayley(m.Ys[i],m.Zs[i])
            X = findX(m.Hs[i], Ls'*Ls)
            F = findF(X, Ls'*Ls)
            LF = Array(cholesky(F).U)
            n_u = size(Ls,1)
            Chat = r2 * inv(Γ) * U' * LF
            Ls = r2 * V * Γ
            K = Chat2K(Chat,m.l[i])
            u2 = [zeros(m.l[i]-1,1,n_u,N2); u]
            u = NNlib.conv(u2,K; pad = 0)
            u = m.nl.(u)
            u = MeanPool((2,1))(u)
            N = Int(N/2)

        elseif L_c <= i <= L # fully connected layers
            if i == L_c
                u = Flux.flatten(permutedims(u, [3, 2, 1, 4])) # Other than Flux.flatten, we flatten by stacking up the signals time point by time point, not channel by chnnel
                Id = Matrix(I,N,N)
                Ls = kron(Id,Ls)
            end
            if i < L
                Γ = Diagonal(m.ds[i])
                U,V = Cayley(m.Ys[i],m.Zs[i])
                W = r2 * inv(Γ) * U' * Ls
                Ls = r2 * V * Γ
                u = m.nl.(W * u .+ m.bs[i] )
            elseif i == L 
                U,V = Cayley(m.Ys[i],m.Zs[i])
                W = U' * Ls
                y = W * u .+ m.bs[i]
            end
        end
    end
    return y
end