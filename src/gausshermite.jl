using StaticArrays
using GaussQuadrature: hermite
using StatsFuns: sqrt2, sqrt2π

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
    lognormaldens::SVector{K,Float64}
end
function GaussHermiteNormalized(k::Int)
    x, w = hermite(k)
    isodd(k) && (x[(k + 1) >> 1] = 0.)
    GaussHermiteNormalized{k}(
        SVector{k,Float64}(x .* sqrt2),
        SVector{k,Float64}(normalize!(w, 1)),
        SVector{k,Float64}((-log(sqrt2π)) .- abs2.(x)))
end
GaussHermiteNormalized(k) = GaussHermiteNormalized(Int(k))

"""
    GHnormd

Memoized values of `GHnorm`{@ref} stored as a `Dict{Int,GaussHermiteNormalized}`
"""
const GHnormd = Dict{Int,GaussHermiteNormalized}(
    0 => GaussHermiteNormalized(SVector{0,Float64}(), SVector{0,Float64}(), SVector{0,Float64}()),
    1 => GaussHermiteNormalized(SVector{1,Float64}(0),SVector{1,Float64}(1),SVector{1,Float64}(-log(sqrt2π)))
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
    // function used below in glmerAGQ
    //
    // fac: mapped integer vector indicating the factor levels
    // u: current conditional modes
    // devRes: current deviance residuals (i.e. similar to results of 
    // family()$dev.resid, but computed in glmFamily.cpp)
    static Ar1 devcCol(const MiVec& fac, const Ar1& u, const Ar1& devRes) {
        Ar1  ans(u.square());
        for (int i = 0; i < devRes.size(); ++i) ans[fac[i] - 1] += devRes[i];
        // return: vector the size of u (i.e. length = number of
        // grouping factor levels), containing the squared conditional
        // modes plus the sum of the deviance residuals associated
        // with each level
        return ans;
    }
=#
function devcCol!(u::Vector{T}, devRes::Vector{T}, refs::Vector{<:Integer}) where T <: AbstractFloat
    map!(abs2, u, u)
    for i in eachindex(refs,devRes)
        u[refs[i]] += devRes[i]
    end
    u
end
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
function AGQDeviance(m::GeneralizedLinearMixedModel, k::Integer)
    length(m.u) == 1 && size(m.u[1], 1) == 1 || throw(ArgumentError("m must have only one scalar random effects term"))
    trm1 = m.LMM.trms[1]
    isa(trm1, ScalarFactorReTerm) || throw(ArgumentError("first term in m must be a ScalarFactorReTerm"))
    u = vec(m.u[1])
    u₀ = vec(m.u₀[1])
    Compat.copyto!(u₀, u)
    devresid = m.resp.devresid
    refs = trm1.f.refs
    devc0 = devcCol!(copy(u), devresid, refs)
    sd = inv.(m.LMM.L.data[Block(1,1)].diag)
    mult = zeros(sd)
    quad = GHnorm(k)
    for (z, wt, ldens) in zip(quad.z, quad.wt, quad.lognormaldens)
        if iszero(z)
            mult .+= wt
        else
            u .= u₀ .+ z .* sd
            updateη!(m)
        end
    end
    mult
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
