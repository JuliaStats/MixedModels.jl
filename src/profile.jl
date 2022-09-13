struct FeProfile{T<:AbstractFloat}
    m::LinearMixedModel{T}
    y₀::Matrix{T}
    xⱼ::Matrix{T}
    j::Integer
end
function FeProfile(m::LinearMixedModel, j::Integer)
    Xy = m.Xymat.xy
    xcols = collect(axes(Xy, 2))
    ycol = pop!(xcols)
    y₀ = Xy[:, ycol:ycol]
    xⱼ = Xy[:, j:j]
    notj = deleteat!(xcols, j)   # indirectly checks range of j
    feterm = FeTerm(Xy[:, notj], m.feterm.cnames[notj])
    return FeProfile(
        fit!(LinearMixedModel(y₀ - xⱼ*m.β[j:j], feterm, m.reterms, m.formula)),
        y₀,
        xⱼ,
        j,
    )
end

function refit!(pr::FeProfile, βⱼ)
    refit!(pr.m, vec(pr.y₀ - pr.xⱼ*βⱼ))
    pr
end

getprops(pr::FeProfile, props=(:objective, :σ, :β, :θ)) = getprops(pr.m, props)

function getprops(m::MixedModel, props=(:objective, :σ, :β, :θ))
    NamedTuple{props}(map(Base.Fix1(getproperty, m), props))
end

function profileβ(m::LinearMixedModel{T}, steps=-5:5) where {T}
    refit!(m)
    β, θ, std, obj = m.β, m.θ, m.stderror, objective(m)
    k = length(θ)
    p = length(β)
    val = map(eachindex(β)) do j
        prj = FeProfile(m, j)
        estj, stdj = β[j], std[j]
        Table(
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
        )
    end
    updateL!(setθ!(m, θ))
    val
end
