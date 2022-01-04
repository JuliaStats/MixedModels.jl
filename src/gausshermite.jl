"""
    GaussHermiteQuadrature

As described in

* [Gauss-Hermite quadrature on Wikipedia](http://en.wikipedia.org/wiki/Gauss-Hermite_quadrature)

*Gauss-Hermite* quadrature uses a weighted sum of values of `f(x)` at specific `x` values to approximate

```math
\\int_{-\\infty}^\\infty f(x) e^{-x^2} dx
```

An `n`-point rule, as returned by `hermite(n)` from the
[`GaussQuadrature``](https://github.com/billmclean/GaussQuadrature.jl) package provides `n` abscicca
values (i.e. values of `x`) and `n` weights.

As noted in the Wikipedia article, a modified version can be used to evaluate the expectation `E[h(x)]`
with respect to a `Normal(μ, σ)` density as
```julia
using MixedModels

gn5 = GHnorm(5)
μ = 3.
σ = 2.
sum(@. abs2(σ*gn5.z + μ)*gn5.w) # E[X^2] where X ∼ N(μ, σ)
```

For evaluation of the log-likelihood of a GLMM the integral to evaluate for each level of
the grouping factor is approximately Gaussian shaped.
"""

"""
    GaussHermiteNormalized{K}

A struct with 2 SVector{K,Float64} members
- `z`: abscissae for the K-point Gauss-Hermite quadrature rule on the Z scale
- `wt`: Gauss-Hermite weights normalized to sum to unity
"""
struct GaussHermiteNormalized{K}
    z::SVector{K,Float64}
    w::SVector{K,Float64}
end
function GaussHermiteNormalized(k::Integer)
    ev = eigen(SymTridiagonal(zeros(k), sqrt.(1:(k - 1))))
    w = abs2.(ev.vectors[1, :])
    return GaussHermiteNormalized(
        SVector{k}((ev.values .- reverse(ev.values)) ./ 2),
        SVector{k}(LinearAlgebra.normalize((w .+ reverse(w)) ./ 2, 1)),
    )
end

function Base.iterate(g::GaussHermiteNormalized{K}, i=1) where {K}
    return (K < i ? nothing : ((z=g.z[i], w=g.w[i]), i + 1))
end

Base.length(g::GaussHermiteNormalized{K}) where {K} = K

"""
    GHnormd

Memoized values of `GHnorm`{@ref} stored as a `Dict{Int,GaussHermiteNormalized}`
"""
const GHnormd = Dict{Int,GaussHermiteNormalized}(
    1 => GaussHermiteNormalized(SVector{1}(0.0), SVector{1}(1.0)),
    2 => GaussHermiteNormalized(SVector{2}(-1.0, 1.0), SVector{2}(0.5, 0.5)),
    3 => GaussHermiteNormalized(
        SVector{3}(-sqrt(3), 0.0, sqrt(3)), SVector{3}(1 / 6, 2 / 3, 1 / 6)
    ),
)

"""
    GHnorm(k::Int)

Return the (unique) GaussHermiteNormalized{k} object.

The function values are stored (memoized) when first evaluated.  Subsequent evaluations
for the same `k` have very low overhead.
"""
GHnorm(k::Int) =
    get!(GHnormd, k) do
        GaussHermiteNormalized(k)
    end
GHnorm(k) = GHnorm(Int(k))
