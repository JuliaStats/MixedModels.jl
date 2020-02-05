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

function refit!(pr::FeProfile, βⱼ)
    refit!(pr.m, pr.y₀ .- βⱼ * pr.xⱼ)
    pr
end

getprops(pr::FeProfile, props=(:objective, :σ, :β, :θ)) = getprops(pr.m, props)

function getprops(m::MixedModel, props=(:objective, :σ, :β, :θ))
    NamedTuple{props}(map(Base.Fix1(getproperty, m), props))
end

function profileβ(m::LinearMixedModel{T}, steps=-5:5) where {T}
    refit!(m)
    β, θ, std, obj, k = m.β, m.θ, m.stderror, objective(m), nθ(m)
    p = length(β)
    val = map(eachindex(β)) do j
        prj = FeProfile(m, j)
        estj, stdj = β[j], std[j]
        map(steps) do s
            if iszero(s)
                (ζ=zero(T), σ=m.σ, β=SVector{p}(β), θ=SVector{k}(θ))
            else
                βj = estj + s * stdj
                props = getprops(refit!(prj, βj))
                splice!(props.β, j:j-1, βj)
                (
                    ζ=sign(s) * sqrt(props.objective - obj),
                    σ=props.σ,
                    β=SVector{p}(props.β),
                    θ=SVector{k}(props.θ),
                )
            end                
        end
    end
    updateL!(setθ!(m, θ))
    val
end
