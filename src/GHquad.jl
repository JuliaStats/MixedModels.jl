using Compat
using GaussQuadrature: hermite
using StatsFuns: sqrt2, log2π

```
    GHmats

Cached matrices providing `k`-point Gauss-Hermite quadrature rules
normalized for mean and standard deviations.

The first column is the absciccae on the `Z` scale 
(i.e. multiples of the standard deviation); the second column is the
weights normalized to sum to 1; and the third column is the log of the
standard normal density at the Z values.
```
const GHmats = Dict{Int,Matrix{Float64}}(
    0 => Matrix{Float64}(Compat.undef, 0, 3), 
    1 => [0.0 1.0 -log2π/2],
    2 => [-1.0 0.5 -log2π/2 - 0.5; 1.0 0.5 -log2π/2 - 0.5;]
)

```
    GHmat(k)

Returns `GHmats`(@ref)[k], creating and caching it if needed.
```
function GHmat(k::Int)
    haskey(GHmats, k) && return GHmats[k]
    x, wt = hermite(k)
    z = sqrt2 .* x    # normalized abscicca (on Z scale)
    isodd(k) && (z[(k + 1) >> 1] = 0.0)
    m = hcat(z, normalize(wt, 1), (-log2π/2) .- abs2.(z) ./ 2)
    GHmats[k] = m
    m
end
GHmat(k) = GHmat(Int(k))
