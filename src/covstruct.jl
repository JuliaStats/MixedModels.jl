"""
    CovarianceStructure{T}

Abstract type for parameterizations of the relative covariance factor `λ` of a
random-effects term.

A concrete subtype defines a map from a parameter vector `θ` to the values of
`λ`, via [`updateλ!`](@ref). For [`Unstructured`](@ref), this map is the
identity map onto the potential nonzeros of `λ` (the classical parameterization
by the elements of the lower triangle) and `θ` is read directly from `λ`. For
all other structures, the map may be nonlinear and the current value of `θ` is
stored in the structure itself.

Each subtype other than `Unstructured` must define:

- `updateλ!(cs, λ, θ)`: write the factor corresponding to `θ` into `λ`
- `canonicalize!(cs)`: map the stored `θ` into the canonical region
- `initialθ(cs)`: the value of `θ` for which `λλ' = I`
- `lowerbd(cs)`: elementwise lower bounds of the canonical region
- `LinearAlgebra.copy_oftype(cs, T)`: convert to a different element type

The parameterizations are designed so that the θ-domain is all of `ℝ^nθ`, with
every `θ` producing a positive semidefinite `λλ'`, singular boundaries attained
at finite `θ`, and the canonical region an elementwise box `θ .≥ lowerbd(cs)`.
This preserves the package's convention of unconstrained optimization followed
by canonicalization ([`rectify!`](@ref)) and the elementwise interpretation of
`lowerbd` used by `issingular` and profiling.
"""
abstract type CovarianceStructure{T} end

"""
    ZeroCorrStruct

Tag type used to route `structure!` (and thus `zerocorr`) to `zerocorr!`.

Zero-correlation terms are represented by an [`Unstructured`](@ref) covariance
structure with a `Diagonal` `λ` and a restricted set of free indices rather
than by a dedicated `CovarianceStructure` subtype, so this tag is not itself a
`CovarianceStructure`.
"""
struct ZeroCorrStruct end

"""
    Unstructured{T} <: CovarianceStructure{T}

The trivial covariance structure: `θ` consists of the potential nonzeros of `λ`
(the lower triangle for a general term, the diagonal for a `zerocorr` term) and
is scattered directly into `λ` via the `inds` field of the `ReMat`.
"""
struct Unstructured{T} <: CovarianceStructure{T} end

"""
    ScaledIdentity{T} <: CovarianceStructure{T}

Scaled-identity ("homogeneous diagonal") covariance structure: `λ = aI`, so
that the covariance matrix of the random effects is `σ²a²I`.

`θ = [a]` with canonical region `a ≥ 0`.
"""
struct ScaledIdentity{T} <: CovarianceStructure{T}
    θ::Vector{T}
end

ScaledIdentity{T}(S::Integer) where {T} = ScaledIdentity{T}(ones(T, 1))

"""
    CompoundSymmetry{T,Het} <: CovarianceStructure{T}

Compound-symmetric covariance structure, homogeneous (`Het == false`) or
heterogeneous (`Het == true`) in the variances.

# Homogeneous case (`HomCS`)

`θ = [a, g]`, where `a` and `g` are the magnitudes of the two distinct singular
values of `λ`: `g` on the one-vector and `a` on its orthogonal complement, i.e.

    λλ' = a²(I - J/S) + g²(J/S)

with `J` the all-ones matrix. All variances equal `((S-1)a² + g²)/S` (relative
to `σ²`) and the common correlation is `ρ = (g² - a²)/((S-1)a² + g²)`. As
`(a, g)` ranges over `ℝ²` this covers exactly the positive-semidefinite
homogeneous compound-symmetric family: `g = 0` gives `ρ = -1/(S-1)`, `a = 0`
gives `ρ = 1`, and `a = g` gives `ρ = 0`. The canonical region is
`a ≥ 0, g ≥ 0`.

# Heterogeneous case (`HetCS`)

`θ = [d₁, …, d_S, b]`, where the `dᵢ` are the relative standard deviations and
`b` parameterizes the common correlation via `c = 2b + Sb²` and
`ρ = c/(1 + c)`, so that

    λλ' = D R D,  D = Diagonal(d),  R = (I + cJ)/(1 + c).

As `b` ranges over `ℝ`, `c` covers `[-1/S, ∞)` and `ρ` covers
`[-1/(S-1), 1)`, with `ρ = 0` at `b = 0`. The canonical region is
`dᵢ ≥ 0, b ≥ -1/S` (the map `b ↦ c` is symmetric about `b = -1/S`). Note that
`ρ = 1` is attained only in the limit `b → ∞`; the common singular boundary
`ρ = -1/(S-1)` is at the finite value `b = -1/S`.
"""
struct CompoundSymmetry{T,Het} <: CovarianceStructure{T}
    θ::Vector{T}
end

const HomCS{T} = CompoundSymmetry{T,false}
const HetCS{T} = CompoundSymmetry{T,true}

function CompoundSymmetry{T,Het}(S::Integer) where {T,Het}
    θ = Het ? vcat(ones(T, Int(S)), zero(T)) : ones(T, 2)
    return CompoundSymmetry{T,Het}(θ)
end

nθ(cs::CovarianceStructure) = length(cs.θ)

"""
    initialθ(cs::CovarianceStructure{T})

Return the initial value of `θ` for `cs`, chosen such that `λλ' = I`.
"""
initialθ(cs::ScaledIdentity{T}) where {T} = ones(T, 1)
initialθ(cs::HomCS{T}) where {T} = ones(T, 2)
initialθ(cs::HetCS{T}) where {T} = vcat(ones(T, length(cs.θ) - 1), zero(T))

lowerbd(cs::ScaledIdentity{T}) where {T} = zeros(T, 1)
lowerbd(cs::HomCS{T}) where {T} = zeros(T, 2)
function lowerbd(cs::HetCS{T}) where {T}
    S = length(cs.θ) - 1
    return vcat(zeros(T, S), -inv(T(S)))
end

"""
    upperbd(cs::CovarianceStructure{T})

Return the elementwise upper bounds of the canonical region for `θ`.

Currently `+Inf` for all implemented structures; provided as a seam for future
structures with bounded parameters (e.g. autoregressive correlations).
"""
upperbd(cs::CovarianceStructure{T}) where {T} = fill(T(Inf), nθ(cs))

"""
    updateλ!(cs::CovarianceStructure, λ, θ::AbstractVector)

Overwrite `λ` with the relative covariance factor corresponding to `θ` under
the parameterization `cs`.

This function reads only the `θ` argument (never the value stored in `cs`) and
is generic in the element types of `λ` and `θ` so that it can be used with
dual numbers for automatic differentiation.
"""
function updateλ!(::ScaledIdentity, λ::Diagonal, θ::AbstractVector)
    fill!(λ.diag, only(θ))
    return λ
end

function updateλ!(::HomCS, λ::LowerTriangular, θ::AbstractVector)
    S = size(λ, 1)
    a, g = θ
    v = ((S - 1) * abs2(a) + abs2(g)) / S
    c = (abs2(g) - abs2(a)) / S
    _cscholesky!(λ.data, v, c)
    return λ
end

function updateλ!(::HetCS, λ::LowerTriangular, θ::AbstractVector)
    data = λ.data
    S = size(data, 1)
    b = last(θ)
    c = (2 + S * b) * b
    ρ = c / (1 + c)
    _cscholesky!(data, one(ρ), ρ)
    # scale by |dᵢ|: the sign of a standard-deviation parameter must not flip
    # the sign of that row's correlations, or the unconstrained optimizer could
    # leave the compound-symmetric family (mixed-sign correlations) for S ≥ 3
    for j in 1:S, i in j:S
        data[i, j] *= abs(θ[i])
    end
    return λ
end

"""
    _cscholesky!(data::AbstractMatrix, v, c)

Overwrite the lower triangle of `data` with the lower Cholesky factor of the
compound-symmetric matrix with diagonal `v` and off-diagonal `c`.

Each column of the factor has a constant value below the diagonal, giving an
`O(S)`-per-column recursion. Guarded so that positive-semidefinite boundary
cases produce finite results.
"""
function _cscholesky!(data::AbstractMatrix, v, c)
    S = size(data, 1)
    s = zero(c)
    for j in 1:S
        ℓ = sqrt(max(v - s, zero(s)))
        data[j, j] = ℓ
        j == S && break
        m = iszero(ℓ) ? zero(ℓ) : (c - s) / ℓ
        for i in (j + 1):S
            data[i, j] = m
        end
        s += abs2(m)
    end
    return data
end

"""
    canonicalize!(cs::CovarianceStructure)

Map the `θ` stored in `cs` into the canonical region (elementwise
`θ .≥ lowerbd(cs)`) without changing the implied covariance matrix `λλ'`.
"""
function canonicalize!(cs::Union{ScaledIdentity,HomCS})
    map!(abs, cs.θ, cs.θ)
    return cs
end

function canonicalize!(cs::HetCS)
    θ = cs.θ
    S = length(θ) - 1
    for i in 1:S
        θ[i] = abs(θ[i])
    end
    lb = -inv(oftype(θ[end], S))
    if θ[end] < lb
        θ[end] = 2 * lb - θ[end]
    end
    return cs
end

function LinearAlgebra.copy_oftype(::Unstructured, ::Type{T}) where {T}
    return Unstructured{T}()
end

function LinearAlgebra.copy_oftype(cs::ScaledIdentity, ::Type{T}) where {T}
    return ScaledIdentity{T}(convert(Vector{T}, copy(cs.θ)))
end

function LinearAlgebra.copy_oftype(cs::CompoundSymmetry{<:Any,Het}, ::Type{T}) where {T,Het}
    return CompoundSymmetry{T,Het}(convert(Vector{T}, copy(cs.θ)))
end

# comparisons are deliberately agnostic to the element type so that, e.g.,
# a bootstrap can be compared across a change in the stored precision
Base.:(==)(::CovarianceStructure, ::CovarianceStructure) = false
Base.:(==)(::Unstructured, ::Unstructured) = true
Base.:(==)(a::ScaledIdentity, b::ScaledIdentity) = a.θ == b.θ
Base.:(==)(a::CompoundSymmetry{<:Any,Het}, b::CompoundSymmetry{<:Any,Het}) where {Het} =
    a.θ == b.θ

Base.isapprox(::CovarianceStructure, ::CovarianceStructure; kwargs...) = false
Base.isapprox(::Unstructured, ::Unstructured; kwargs...) = true
function Base.isapprox(a::ScaledIdentity, b::ScaledIdentity; kwargs...)
    return isapprox(a.θ, b.θ; kwargs...)
end
function Base.isapprox(
    a::CompoundSymmetry{<:Any,Het}, b::CompoundSymmetry{<:Any,Het}; kwargs...
) where {Het}
    return isapprox(a.θ, b.θ; kwargs...)
end
