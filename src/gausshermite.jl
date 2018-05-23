using StaticArrays
using GaussQuadrature: hermite
using StatsFuns: sqrt2, sqrt2π

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
