using Compat, StaticArrays
using Compat.LinearAlgebra

"""
    Gauss-Hermite

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
using GaussQuadrature

x, w = hermite(5)
μ = 3.
σ = 2.
sum(@. abs2(√2*σ*x + μ)*w)/√π  # E[X^2] where X ∼ N(μ, σ)
```

For evaluation of the log-likelihood of a GLMM the integral to evaluate for each level of the grouping
factor is approximately Gaussian shaped.
"""
GaussHermiteQuadrature
"""
    GaussHermiteNormalized{K}

A struct with 3 SVector{K,Float64} members
- `z`: abscissae for the K-point Gauss-Hermite quadrature rule on the Z scale
- `wt`: Gauss-Hermite weights normalized to sum to unity
- `lognormaldens`: log of standard normal density at `z`
"""
struct GaussHermiteNormalized{K}
    z::SVector{K, Float64}
    wt::SVector{K,Float64}
    logdensity::SVector{K,Float64}
end
function GaussHermiteNormalized(k::Integer)
    ev = eigfact(SymTridiagonal(zeros(k), sqrt.((1:k-1) ./ 2)))
    z = (ev.values .- reverse(ev.values)) ./ √2
    w = abs2.(ev.vectors[1,:])
    GaussHermiteNormalized(SVector{k}(z), 
        SVector{k}((w .+ reverse(w)) ./ 2),
        SVector{k}((-log(2π)/2) .- abs2.(z) ./ 2))
end
@static if VERSION ≥ v"0.7.0-DEV.5124"
    Base.iterate(g::GaussHermiteNormalized{K}, i=1) where {K} = 
        (K < i ? nothing : ((g.z[i], g.wt[i], g.logdensity[i]), i + 1))
else
    Base.start(gh::GaussHermiteNormalized) = 1
    Base.next(gh::GaussHermiteNormalized, i) = (gh.z[i], gh.wt[i], gh.logdensity[i]), i+1
    Base.done(gh::GaussHermiteNormalized{K}, i) where {K} = K < i 
end
"""
    GHnormd

Memoized values of `GHnorm`{@ref} stored as a `Dict{Int,GaussHermiteNormalized}`
"""
const GHnormd = Dict{Int,GaussHermiteNormalized}(
    1 => GaussHermiteNormalized(SVector{1}(0.),SVector{1}(1.),SVector{1}(-log(2π)/2))
    )

"""
    GHnorm(k::Int)

Return the (unique) GaussHermiteNormalized{k} object.

The values are memoized in `GHnormd`{@ref} when first evaluated.  Subsequent evaluations
for the same `k` have very low overhead.
"""
GHnorm(k::Int) = get!(GHnormd, k) do
    GaussHermiteNormalized(k)
end
GHnorm(k) = GHnorm(Int(k))

#=
steps:
1. ensure that the conditional standard deviations of the random effects are being evalated correctly
2. for each value of u, just need to call updateμ! to get the deviance residuals
3. Is it necessary to sum the deviance residuals for each level of the factor separately? Related to separation of integrals?
=#
#=

        pwrssUpdate(rp, pp, true, tol, maxit, verb); // should be a
                                                     // no-op

                    // devc0: vector with one element per grouping
                    // factor level containing the the squared
                    // conditional modes plus the sum of the deviance
                    // residuals associated with each level
        const Ar1      devc0(devcCol(fac, pp->u(1.), rp->devResid())); 
        const unsigned int q(pp->u0().size());
        if (pp->L().factor()->nzmax !=  q)
            throw std::invalid_argument("AGQ only defined for a single scalar random-effects term");
        const Ar1         sd(MAr1((double*)pp->L().factor()->x, q).inverse());
=#
struct RaggedArray{T,I}
    vals::Vector{T}
    inds::Vector{I}
end
function Base.sum!(s::Vector{T}, a::RaggedArray{T}) where T
    for (v, i) in zip(a.vals, a.inds)
        s[i] += v
    end
    s
end
function AGQDeviance(m::GeneralizedLinearMixedModel, k::Integer)
    length(m.u[1]) == length(m.AGQ.devc) || 
        throw(ArgumentError("m must have a single scalar random-effect term"))
    u = vec(m.u[1])
    u₀ = vec(m.u₀[1])
    Compat.copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, m.LMM.trms[1].f.refs)
    devc0 = sum!(broadcast!(abs2, m.AGQ.devc0, u), ra)  # the deviance components at z = 0
    sd = broadcast!(inv, m.AGQ.sd, m.LMM.L.data[Block(1,1)].diag)
    mult = fill!(m.AGQ.mult, 0)
    devc = m.AGQ.devc
    for (z, wt, ldens) in GHnorm(k)
        if iszero(z)
            mult .+= wt
        else
            u .= u₀ .+ z .* sd
            updateη!(m)
            mult .+= exp.(-(sum!(broadcast!(abs2, devc, u), ra) .- devc0) ./ 2 .- ldens) .* (wt/√2π)
        end
    end
    Compat.copyto!(u, u₀)
    updateη!(m)
    sum(devc0) + logdet(m) - 2 * sum(log, mult)
end
#=
        const MMat     GQmat(as<MMat>(GQmat_));
        Ar1             mult(q);

        mult.setZero();
        for (int i = 0; i < GQmat.rows(); ++i) {
            double     zknot(GQmat(i, 0));
            if (zknot == 0)
                mult += Ar1::Constant(q, GQmat(i, 1));
            else {
                pp->setU0(zknot * sd); // to be added to current delu
                rp->updateMu(pp->linPred(1.));
                mult += (-0.5 * (devcCol(fac, pp->u(1.), rp->devResid()) - devc0) -
                         GQmat(i, 2)).exp() * GQmat(i, 1)/sqrt2pi;
            }
        }
        pp->setU0(Vec::Zero(q)); // restore settings from pwrssUpdate;
        rp->updateMu(pp->linPred(1.));
        return ::Rf_ScalarReal(devc0.sum() + pp->ldL2() - 2 * std::log(mult.prod()));

=#
