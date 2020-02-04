struct FeProfile{T<:AbstractFloat}
    m::LinearMixedModel{T}
    y₀::Vector{T}
    xⱼ::Vector{T}
    j::Integer
end

function FeProfile(m::LinearMixedModel, j::Integer)
    allt = m.allterms
    feind = findfirst(Base.Fix2(isa, FeMat), allt)
    Xorig = allt[feind]
    β = m.β
    Xmat = Xorig.x
    n, p = size(Xmat)
    1 ≤ j ≤ p || throw(ArgumentError("1 ≤ j = $j ≤ p = $p is not true"))
    y₀ = copy(response(m))
    xⱼ = vec(Xmat[:, j])
    allterms = allt[1:feind - 1]
    push!(allterms, FeMat(Xorig.x[:, Not(j)], Xorig.cnames[Not(j)]))
    push!(allterms, FeMat(reshape(y₀ .- β[j] * xⱼ, :, 1), [""]))
    A, L = createAL(allterms)
    FeProfile(fit!(LinearMixedModel(m.formula, allterms, m.sqrtwts, A, L, deepcopy(m.optsum))), y₀, xⱼ, j)
end

refit!(pr::FeProfile, βⱼ) = refit!(pr.m, pr.y₀ .- βⱼ * pr.xⱼ)

"""
    dropcol(m::AbstractMatrix, j)

Return a copy of `m` having dropped column `j`
"""
dropcol(M::AbstractMatrix, j) = M[:, deleteat!(collect(axes(M, 2)), j)]

function profileβ(m::LinearMixedModel{T}, β::Vector{T}=m.β, std::Vector{T}=m.stderror) where {T}
    d0 = deviance(m)
    y0 = copy(response(m))
    X0 = copy(first(m.feterms))
    eachβ = eachindex(β)
    colvec = collect(eachβ)
    for i in eachβ
        freecols = deleteat!(colvec, i)
        m.feterms[1] = FeMat(X0.x[:,freecols], X0.cnames[freecols])
    end
end
