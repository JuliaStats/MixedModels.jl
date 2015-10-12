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
    m.opt.feval = m.opt.geval = -1
    m
end

"""
Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space the last column of `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.
"""
function bootstrap(m::LinearMixedModel,N::Integer,saveresults::Function,σ::Real=-1,mods::Vector{LinearMixedModel}=LinearMixedModel[])
    if σ < 0.
        σ = √varest(m)
    end
    trms = m.trms
    nt = length(trms)
    Xy = trms[nt]
    n,pp1 = size(Xy)
    X = sub(Xy,:,1:pp1-1)
    y = sub(Xy,:,pp1)
    y0 = copy(y)
    fev = X*coef(m)
    vfac = [σ*convert(LowerTriangular,λ) for λ in m.Λ]  # lower Cholesky factors of relative covariances
    remats = Matrix{Float64}[]
    for i in eachindex(vfac)
        vsz = size(trms[i],2)
        nr = size(vfac[i],1)
        push!(remats,reshape(zeros(vsz),(nr,div(vsz,nr))))
    end
    for kk in 1:N
        for i in 1:n
            y[i] = fev[i] + σ*randn()
        end
        for j in eachindex(remats)
            mat = vec(A_mul_B!(vfac[j],randn!(remats[j])))
            A_mul_B!(1.,trms[j],vec(mat),1.,y)
        end
        regenerateAend!(m)
        resetθ!(m)
        fit!(m)
        saveresults(kk,m)
    end
    copy!(y,y0)
    regenerateAend!(m)
    resetθ!(m)
    fit!(m)
end
