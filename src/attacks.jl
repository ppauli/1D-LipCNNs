function attack_acc(m,ϵ_vec)

    loss(x, y) = crossentropy(m(x), y)

    # Comparison functions
    compare(y::OneHotMatrix, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
    accuracy(x, y::OneHotMatrix) = mean(compare(y, m(x)))

    acc_adv = zeros(length(ϵ_vec),1)
    Len = length(ϵ_vec)

    for i in 1:Len
        #test_x_adv = FGSM(m, loss, test_x, test_y ; ϵ = ϵ_vec[i]/300)
        test_x_adv = PGD(m, loss, test_x, test_y; ϵ = ϵ_vec[i], step_size = ϵ_vec[i]/10, iters = 10, clamp_range = (0, 1))
        acc_adv[i] = accuracy(test_x_adv,test_y)
    end

    return acc_adv
end


function FGSM(model, loss, x, y; ϵ = 0.1, clamp_range = (0, 1))
    J = gradient(() -> loss(x, y), params(x))
    x_adv = clamp.(x + (Float32(ϵ) * sign.(J[x])), clamp_range...)
    return x_adv
end

function PGD(model, loss, x, y; ϵ = 10, step_size = 0.001, iters = 20, clamp_range = (0, 1))
    x_adv = clamp.(x + (gpu(randn(Float32, size(x)...)) * Float32(step_size)), clamp_range...); # start from the random point
    δ = norm(x - x_adv)
    if δ > ϵ
        x_adv = x + (x_adv-x)*ϵ/δ
    end
    iter = 1; while  iter <= iters
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = norm(x - x_adv)
        if δ > ϵ
            x_adv = x + (x_adv-x)*ϵ/δ
        end
        iter += 1
    end
    return x_adv
end