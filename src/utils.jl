function LipCNN2Weights(m,channels,fc_layers, maxpoolind)
l = 3
nu = 1
ny = 5
ρ = m[1].γ
Lc = length(channels)
L = length(fc_layers) + length(channels)

if maxpoolind == 1
    Lm = ρ*Matrix(I,nu,nu)
    # Set up model
    m2 = Chain(
        # First convolution, operating upon a 128x1 signal
        Conv((3, 1), channels[1]=>channels[2], pad=(2,0), relu),
        x -> maxpool(x[1:128,:,:,:], (2,1)),
    
        # Second convolution, operating upon a 64x1 signal
        Conv((3, 1), channels[2]=>channels[3], pad=(2,0), relu),
        x -> maxpool(x[1:64,:,:,:], (2,1)),
    
        # Reshape 3d tensor into a 2d one, at this point it should be (128, 1, 3, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> permutedims(x, [3, 2, 1, 4]),
        x -> reshape(x, :, size(x, 4)),
        Dense(96, fc_layers[1], relu),
        Dense(fc_layers[1], 5)
    
        # Finally, softmax to get nice probabilities
        #softmax,
    )
else
    Lm = 2*ρ*Matrix(I,nu,nu)
    # Set up model
    m2 = Chain(
        # First convolution, operating upon a 128x1 signal
        Conv((3, 1), channels[1]=>channels[2], pad=(2,0), relu),
        x -> meanpool(x[1:128,:,:,:], (2,1)),
    
        # Second convolution, operating upon a 64x1 signal
        Conv((3, 1), channels[2]=>channels[3], pad=(2,0), relu),
        x -> meanpool(x[1:64,:,:,:], (2,1)),
    
        # Reshape 3d tensor into a 2d one, at this point it should be (128, 1, 3, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> permutedims(x, [3, 2, 1, 4]),
        x -> reshape(x, :, size(x, 4)),
        Dense(96, fc_layers[1], relu),
        Dense(fc_layers[1], 5)
    
        # Finally, softmax to get nice probabilities
        #softmax,
    )
end

for ii = 1:L
    if ii < Lc
        Y = m[1].Ys[ii]
        Z = m[1].Zs[ii]
        H = m[1].Hs[ii]
        b = m[1].bs[ii]
        γ = m[1].ds[ii]
        l = m[1].l[ii]
        if maxpoolind == 1
            ls = m[1].ls[ii]
            K,b,Ls = LipCNNmax2CNN(Y,Z,H,γ,ls,b,l,Lm)
        else
            K,b,Ls = LipCNN2CNN(Y,Z,H,γ,b,l,Lm)
        end
        Flux.params(m2)[2*(ii-1)+1] .= K
        Flux.params(m2)[2*ii] .= b
        Lm = Ls
    elseif Lc <= ii < L
        Y = m[1].Ys[ii]
        Z = m[1].Zs[ii]
        b = m[1].bs[ii]
        γ = m[1].ds[ii]
        if ii == Lc
            N = Int(128/4)
            Id = Matrix(I,N,N)
            Lm = kron(Id,Lm)
        end
        W, b, Ls = LipCNN2FNN(Y,Z,γ,b,Lm,0)
        Flux.params(m2)[2*(ii-1)+1] .= W
        Flux.params(m2)[2*ii] .= b
        Lm = Ls
    elseif ii == L
        Y = m[1].Ys[ii]
        Z = m[1].Zs[ii]
        b = m[1].bs[ii]
        W, b, Ls = LipCNN2FNN(Y,Z,ones(ny),b,Lm,1)
        Flux.params(m2)[2*(ii-1)+1] .= W
        Flux.params(m2)[2*ii] .= b
    end
end
    return m2
end

function LooseSoftmax(m,maxpoolind)
    if maxpoolind == 1
        # Set up model
        m2 = Chain(
            # First convolution, operating upon a 128x1 signal
            Conv((3, 1), channels[1]=>channels[2], pad=(2,0), relu),
            x -> maxpool(x[1:128,:,:,:], (2,1)),
        
            # Second convolution, operating upon a 64x1 signal
            Conv((3, 1), channels[2]=>channels[3], pad=(2,0), relu),
            x -> maxpool(x[1:64,:,:,:], (2,1)),
        
            # Reshape 3d tensor into a 2d one, at this point it should be (128, 1, 3, N)
            # which is where we get the 288 in the `Dense` layer below:
            x -> permutedims(x, [3, 2, 1, 4]),
            x -> reshape(x, :, size(x, 4)),
            Dense(96, fc_layers[1], relu),
            Dense(fc_layers[1], 5)
        
            # Finally, softmax to get nice probabilities
            #softmax,
        )
    else
        # Set up model
        m2 = Chain(
            # First convolution, operating upon a 128x1 signal
            Conv((3, 1), channels[1]=>channels[2], pad=(2,0), relu),
            x -> meanpool(x[1:128,:,:,:], (2,1)),
        
            # Second convolution, operating upon a 64x1 signal
            Conv((3, 1), channels[2]=>channels[3], pad=(2,0), relu),
            x -> meanpool(x[1:64,:,:,:], (2,1)),
        
            # Reshape 3d tensor into a 2d one, at this point it should be (128, 1, 3, N)
            # which is where we get the 288 in the `Dense` layer below:
            x -> permutedims(x, [3, 2, 1, 4]),
            x -> reshape(x, :, size(x, 4)),
            Dense(96, fc_layers[1], relu),
            Dense(fc_layers[1], 5)
        
            # Finally, softmax to get nice probabilities
            #softmax,
        )
    end
    for ii = 1:Int(length(Flux.params(m))/2)
        Flux.params(m2)[2*(ii-1)+1] .= Flux.params(m)[2*(ii-1)+1]
        Flux.params(m2)[2*ii] .= Flux.params(m)[2*ii]
    end
return m2
end


function findX(H,Qm)
    n_u = size(Qm,1)
    n_x = size(H,1)
    ϵ = 0.000001
    X = zeros(n_x,n_x)
    A, B = construct_AB(n_x,n_u)
    for k = 0:n_x-n_u
        X = X + A^k*(B*inv(Qm)*B'+H'*H+ϵ*Matrix(I,n_x,n_x))*(A')^k
    end
    return X
end

function findF(X,Qm)
    n_u = size(Qm,1)
    n_x = size(X,1)
    P = inv(X)
    A, B = construct_AB(n_x,n_u)
    F = [P-A'*P*A -A'*P*B; -B'*P*A Qm.-B'*P*B]
    return 0.5*(F+F')
end

function construct_AB(n_x,n_u)
    A = [zeros(n_x-n_u,n_u) Matrix(I,n_x-n_u,n_x-n_u); zeros(n_u,n_x)]
    B = [zeros(n_x-n_u,n_u); Matrix(I,n_u,n_u)]
    return A, B
end

function create_ss(K)
    l,~,ci,co = size(K)
    A,B = construct_AB((l-1)*ci,ci)
    C = zeros(co, (l-1)*ci)
    for i = 1:l-1
        C[:,(i-1)*ci+1:i*ci] = reshape(K[l-i+1,:,:,:], ci,co)'
    end
    D::Matrix{Float64} = reshape(K[1,:,:,:], ci,co)'

    return A, B, C, D
end

function Cayley(Y,Z)
    M = Y' * Y + Z - Z'
    V = inv(I+M)*(I-M)
    U = 2*Y*inv(I+M)
    return U,V
end

function Chat2K(Chat,l)
    ny,nxu = size(Chat)
    nu = Int(nxu/l) 
    K = reshape(Chat,ny,nu,l) # kernel, cin, cout
    K2 = reverse(K,dims = 3)
    K3 = permutedims(K2, [3, 2, 1])
    K4 = reshape(K3,size(K3,1),1,size(K3,2),size(K3,3))
    return K4
end

function LipCNN2CNN(Y,Z,H,γ,b,l,Lm)
    Qm = Lm'*Lm
    X = findX(H, Qm)
    F = findF(X, Qm)
    LF = Array(cholesky(F).U)
    
    U,V = Cayley(Y, Z)
    Chat = sqrt(2) * inv(Diagonal(γ)) * U' * LF
    ci = size(Qm,1)
    co = size(Chat,1)
    K = Chat2K(Chat, l)
    K = reshape(K,l,1,ci,co)
    L = sqrt(2) * V * Diagonal(γ) 

    return K, b, L, Chat
end

function LipCNNmax2CNN(Y,Z,H,γt,ls,b,l,Lm)
    Qm = Lm'*Lm
    X = findX(H, Qm)
    F = findF(X, Qm)
    LF = Array(cholesky(F).U)
    Ls = Diagonal(ls)
    Γt = Diagonal(γt)
    Λ = 0.5*(Γt.^2 .+ Ls.^2)
    
    U,V = Cayley(Y, Z)
    Chat = sqrt(2) * inv(Λ) * Γt * [V; U]' * LF
    ci = size(Qm,1)
    co = size(Chat,1)
    K = Chat2K(Chat, l)
    K = reshape(K,l,1,ci,co)

    return K, b, Ls, Chat
end

function LipCNN2FNN(Y,Z,γ,b,Lm,last)
    U,V = Cayley(Y, Z)
    if last == 1
        W = U' * Lm
        L = V
    else
        W = sqrt(2) * inv(Diagonal(γ)) * U' * Lm
        L = sqrt(2) * V * Diagonal(γ)        
    end
    return W, b, L
end

function load_data()
    fid_train, fid_test = h5open("data\\train_ecg.hdf5", "r"), h5open("data\\test_ecg.hdf5", "r") 

    train_x, train_y = read(fid_train,"x_train"), read(fid_train,"y_train")
    test_x, test_y = read(fid_test,"x_test"), read(fid_test,"y_test")

    # Reshape as appropriate for training
    train_y, test_y = onehotbatch(train_y, 0:4), onehotbatch(test_y, 0:4)
    train_x, test_x = reshape(train_x,(128,1,1,size(train_y,2))), reshape(test_x,(128,1,1,size(test_y,2)))
    return train_x, train_y, test_x, test_y
end


function vec2str(vec)
    if length(vec)>1
        str_tmp = string.((vec[1:end-1],',')...)
        str = string.(str_tmp...,vec[end])
    elseif length(vec) == 1
        str = string.(vec...)
    else 
        str = " "
    end
    return str
end