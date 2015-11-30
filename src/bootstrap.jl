"""
Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space the last column of `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.
"""
function bootstrap(m::LinearMixedModel,N::Integer,saveresults::Function)
    y0 = copy(model_response(m))
    β = coef(m)
    σ = sdest(m)
    θ = m[:θ]
    for i in 1:N
        saveresults(i,simulate!(m;β=β,σ=σ,θ=θ))
    end
    refit!(m,y0)
end


"""
Regenerate the last column of `m.A` from `m.trms`

This should be called after updating parts of `m.trms[end]`, typically the response.
"""
function regenerateAend!(m::LinearMixedModel)
    n = Base.LinAlg.chksquare(m.A)
    trmn = m.trms[n]
    for i in 1:n
        Ac_mul_B!(m.A[i,n],m.trms[i],trmn)
    end
    m
end

"""
Reset the value of `m.θ` to the initial values
"""
function resetθ!(m::LinearMixedModel)
    m[:θ] = m.opt.initial
    m.opt.feval = -1
    m.opt.fmin = Inf
    m
end

"""
Add unscaled random effects to y
"""
function unscaledre!(y::AbstractVector,M::ScalarReMat,L::LowerTriangular)
    z = M.z
    length(y) == length(z) && size(L) == (1,1) || throw(DimensionMismatch())
    re = L[1,1]*randn(length(M.f.pool))
    inds = M.f.refs
    for i in eachindex(y)
        y[i] += re[inds[i]]*z[i]
    end
    y
end
function unscaledre!(y::AbstractVector,M::VectorReMat,L::LowerTriangular)
    Z = M.z
    length(y) == size(Z,2) || throw(DimensionMismatch())
    re = A_mul_B!(L,randn(size(Z,1),length(M.f.pool)))
    inds = M.f.refs
    for i in eachindex(y)
        y[i] += dot(sub(Z,:,i),sub(re,:,Int(inds[i])))
    end
    y
end

"""
Simulate a response vector from model `m`, and refit `m`.

- m, LinearMixedModel.
- β, fixed effects parameter vector
- σ, standard deviation of the per-observation random noise term
- σv, vector of standard deviations for the scalar random effects.
"""
function simulate!(m::LinearMixedModel;β=coef(m),σ=sdest(m),θ=m[:θ])
    m[:θ] = θ        # side-effect of checking for correct length(θ)
    trms = m.trms
    Xy = trms[end] # hcat of fixed-effects model matrix X and response y
    pp1 = size(Xy,2)
    Λ = m.Λ
    y = randn!(sub(Xy,:,pp1)) # initialize to standard normal noise
    for j in eachindex(Λ)     # add the unscaled random effects
        unscaledre!(y,trms[j],Λ[j])
    end
    Base.LinAlg.BLAS.gemv!('N',1.0,sub(Xy,:,1:pp1-1),β,σ,y)
    m |> regenerateAend! |> resetθ! |> fit!
end

"""
refit the model `m` with response `y`
"""
function refit!(m::LinearMixedModel,y)
    copy!(model_response(m),y)
    m |> regenerateAend! |> resetθ! |> fit!
end

"""
extract the response (as a reference)

In Julia 0.5 this can be a one-liner `m.trms[end][:,end]`
"""
function StatsBase.model_response(m::LinearMixedModel)
    Xy = m.trms[end]
    sub(Xy,:,size(Xy,2))
end
