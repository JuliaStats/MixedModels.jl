
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
