"""
    MatrixTerm

Term with an explicit, constant matrix representation

#Members
* `x`: matrix
* `wtx`: weighted matrix
* `cnames`: vector of column names
"""
mutable struct MatrixTerm{T,S<:AbstractMatrix} <: AbstractTerm{T}
    x::S
    wtx::S
    cnames::Vector{String}
end
MatrixTerm(X, cnms) = MatrixTerm{eltype(X),typeof(X)}(X, X, cnms)
function MatrixTerm(y::Vector)
    T = eltype(y)
    m = reshape(y, (length(y), 1))
    MatrixTerm{T,Matrix{T}}(m, m, [""])
end

function reweight!(A::MatrixTerm{T}, sqrtwts::Vector{T}) where T
    if !isempty(sqrtwts)
        if (A.x === A.wtx)
            A.wtx = similar(A.x)
        end
        scale!(A.wtx, sqrtwts, A.x)
    end
    A
end

eltype(A::MatrixTerm) = eltype(A.wtx)

Base.size(A::MatrixTerm) = size(A.wtx)

Base.size(A::MatrixTerm, i) = size(A.wtx, i)

Base.copy!(A::MatrixTerm{T}, src::AbstractVecOrMat{T}) where {T} = copy!(A.x, src)

Ac_mul_B!(R::AbstractMatrix{T}, A::MatrixTerm{T}, B::MatrixTerm{T}) where {T} =
    Ac_mul_B!(R, A.wtx, B.wtx)

Base.Ac_mul_B(A::MatrixTerm{T}, B::MatrixTerm{T}) where {T} = Ac_mul_B(A.wtx, B.wtx)

A_mul_B!(R::StridedVecOrMat{T}, A::MatrixTerm{T}, B::StridedVecOrMat{T}) where {T} =
    A_mul_B!(R, A.x, B)

const AbstractFactor{V,R} =
    Union{NullableCategoricalVector{V,R},CategoricalVector{V,R},PooledDataVector{V,R}}

"""
    asfactor(f)

Return `f` as a AbstractFactor.

This function and the `AbstractFactor` union can be removed once `CategoricalArrays` replace
`PooledDataArray`
"""
asfactor(f::AbstractFactor) = f
asfactor(f) = pool(f)

## FIXME: Create AbstractFactorReTerm, ScalarFactorReTerm and VectorFactorReTerm tyoes
## Advantage is to use dispatch for methods in linalg/lambdaproducts.jl that currently
## use short cuts based on vsize(arg) == 1

"""
    FactorReTerm

Random-effects term from a grouping factor, model matrix and block pattern

# Members
* `f`: the grouping factor as an `AbstractFactor`
* `z`: the transposed raw random-effects model matrix
* `wtz`: a weighted copy of `z`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
* `blks`: a `Vector{Int}` of block sizes within `Λ`
* `Λ`: the relative covariance factor
* `inds`: linear indices of θ elements in the relative covariance factor
"""
mutable struct FactorReTerm{T<:AbstractFloat,V,R} <: AbstractTerm{T}
    f::AbstractFactor{V,R}
    z::Matrix{T}
    wtz::Matrix{T}
    fnm::Symbol
    cnms::Vector{String}
    blks::Vector{Int}
    Λ::Matrix{T}
    inds::Vector{Int}
end
function FactorReTerm(f::AbstractFactor, z::Matrix, fnm, cnms, blks)
    @argcheck (n = sum(blks)) == size(z, 1) DimensionMisMatch
    m = reshape(1:abs2(n), (n, n))
    offset = 0
    inds = sizehint!(Int[], (n * (n + 1)) >> 1)
    for k in blks
        for j in 1:k, i in j:k
            push!(inds, m[offset + i, offset + j])
        end
        offset += k
    end
    FactorReTerm(f, z, z, fnm, cnms, blks, eye(eltype(z), n), inds)
end
# convenience constructor for testing
FactorReTerm(f::AbstractFactor) =
    FactorReTerm(f, ones(1, length(f)), :G, ["(Intercept)"], [1])

function reweight!(A::FactorReTerm, sqrtwts::Vector)
    if !isempty(sqrtwts)
        if A.z === A.wtz
            A.wtz = similar(A.z)
        end
        scale!(A.wtz, A.z, sqrtwts)
    end
    A
end

"""
    levs(A::FactorReTerm)

Return the levels of the grouping factor.

This is to disambiguate a call to `levels` as both `DataArrays`
and `CategoricalArrays` export it.
"""
function levs(A::FactorReTerm)
    f = A.f
    isa(f, PooledDataArray) ? DataArrays.levels(f) : CategoricalArrays.levels(f)
end

"""
    nlevs(A::FactorReTerm)

Return the number of levels in the grouping factor of `A`.
"""
nlevs(A::FactorReTerm) = length(levs(A))

"""
    nrandomeff(A::FactorReTerm)

Return the total number of random effects in A.
"""
nrandomeff(A::FactorReTerm) = nlevs(A) * vsize(A)

"""
    vsize(A::FactorReTerm)

Return the size of vector-valued random effects.
"""
vsize(A::FactorReTerm) = size(A.z, 1)

Base.eltype{T}(R::FactorReTerm{T}) = T

Base.full(R::FactorReTerm) = full(sparse(R))

Base.size(A::FactorReTerm) = (length(A.f), nrandomeff(A))

Base.size(A::FactorReTerm, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.f) :  i == 2 ? nrandomeff(A) : 1

"""
    sparse(R::FactorReTerm)

Convert the random effects model matrix `R.z` from the internal, compressed form
to the expanded form.  The (transposed) "compressed" form has one row per data
observation, and one column per random effect.  The "expanded" form has the same
row structure but one column for each random effect × grouping level
combination.
"""
function Base.sparse(R::FactorReTerm)
    zrows, zcols = size(R.z)
    I = convert(Vector{Int32}, repeat(1:zcols, inner=vsize(R)))
    J = vec(Int32[(R.f.refs[j] - 1) * vsize(R) + i for i in 1:zrows, j in 1:zcols])
    sparse(I, J, vec(R.z))
end

cond(A::FactorReTerm) = cond(LowerTriangular(A.Λ))

"""
    nθ(A::FactorReTerm)

Return the number of free parameters in the relative covariance matrix Λ
"""
function nθ end
nθ(A::FactorReTerm) = length(A.inds)
nθ(A::MatrixTerm) = 0

getΛ(A::FactorReTerm) = A.Λ

"""
    getθ!{T}(v::AbstractVector{T}, A::FactorReTerm{T})

Overwrite `v` with the elements of the blocks in the lower triangle of `A.Λ` (column-major ordering)
"""
function getθ! end

function getθ!(v::StridedVector{T}, A::FactorReTerm{T}) where T
    @argcheck(length(v) == length(A.inds), DimensionMismatch)
    inds = A.inds
    m = A.Λ
    @inbounds for i in eachindex(inds)
        v[i] = m[inds[i]]
    end
    v
end
function getθ!(v::StridedVector{T}, A::MatrixTerm{T}) where T
    @argcheck(length(v) == 0, DimensionMisMatch)
    v
end

"""
    getθ(A::FactorReTerm)

Return a vector of the elements of the lower triangle blocks in `A.Λ` (column-major ordering)
"""
function getθ end

getθ(A::FactorReTerm) = A.Λ[A.inds]
getθ(A::MatrixTerm{T}) where {T} = T[]
getθ(v::Vector{AbstractTerm{T}}) where {T} = reduce(append!, T[], getθ(t) for t in v)

"""
    lowerbd{T}(A::FactorReTerm{T})
    lowerbd{T}(A::MatrixTerm{T})
    lowerbd{T}(v::Vector{AbstractTerm{T}})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
function lowerbd end

lowerbd(v::Vector{AbstractTerm{T}}) where {T} = reduce(append!, T[], lowerbd(t) for t in v)
lowerbd(A::FactorReTerm{T}) where {T} =
    T[x ∈ diagind(A.Λ) ? zero(T) : convert(T, -Inf) for x in A.inds]
lowerbd(A::MatrixTerm{T}) where {T} = T[]

function setθ!(A::FactorReTerm{T}, v::AbstractVector{T}) where T
    @argcheck(length(v) == length(A.inds), DimensionMismatch)
    m = A.Λ
    inds = A.inds
    @inbounds for i in eachindex(inds)
        m[inds[i]] = v[i]
    end
    A
end
function setθ!(trms::Vector{AbstractTerm{T}}, v::Vector{T}) where T
    offset = 0
    for trm in trms
        if (k = nθ(trm)) > 0
            setθ!(trm, view(v, (1:k) + offset))
            offset += k
        end
    end
    trms
end

function Ac_mul_B!(α::Real, A::FactorReTerm{T}, B::MatrixTerm{T}, β::Real,
                   R::Matrix{T}) where T
    n, q = size(A)
    Bwt = B.wtx
    k = size(Bwt, 2)
    @argcheck(size(R, 1) == q && size(Bwt, 1) == n && size(R, 2) == k, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = A.f.refs
    Awtz = A.wtz
    l = vsize(A)
    if l == 1
        for j in 1:k, i in 1:n
            R[rr[i], j] += α * Awtz[i] * Bwt[i, j]
        end
    else
        for j in 1:k, i in 1:n
            roffset = (rr[i] - 1) * l
            mul = α * Bwt[i, j]
            for ii in 1 : l
                R[roffset + ii, j] += mul * Awtz[ii, i]
            end
        end
    end
    R
end

Ac_mul_B!(R::Matrix{T}, A::FactorReTerm{T}, B::MatrixTerm{T}) where {T} =
    Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::FactorReTerm{T}, B::MatrixTerm{T}) where T
    Ac_mul_B!(zeros(eltype(B), (size(A, 2), size(B, 2))), A, B)
end

function Ac_mul_B!(α::Real, A::MatrixTerm{T}, B::FactorReTerm{T}, β::Real,
                   R::Matrix{T}) where T
    Awt = A.wtx
    n, p = size(Awt)
    q = size(B, 2)
    @argcheck(size(R, 1) == p && size(B, 1) == n && size(R, 2) == q, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = B.f.refs
    zz = B.wtz
    if vsize(B) == 1
        for i in 1:p, j in 1:n
            R[i, rr[j]] += α * zz[j] * Awt[j, i]
        end
    else
        l = size(zz, 1)
        for j in 1:p, i in 1:n
            roffset = (rr[i] - 1) * l
            mul = α * Awt[i, j]
            for ii in 1:l
                R[j, roffset + ii] += mul * zz[ii, i]
            end
        end
    end
    R
end

Ac_mul_B!(R::Matrix{T}, A::MatrixTerm{T}, B::FactorReTerm{T}) where {T} =
    Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::MatrixTerm{T}, B::FactorReTerm{T}) where T
    Ac_mul_B!(zeros(eltype(B), (size(A, 2), size(B, 2))), A, B)
end

function Ac_mul_B!(C::Diagonal{T}, A::FactorReTerm{T}, B::FactorReTerm{T}) where T
    @argcheck A === B && vsize(A) == 1
    Az = A.wtz
    d = C.diag
    fill!(d, zero(T))
    refs = A.f.refs
    for i in eachindex(refs)
        d[refs[i]] += abs2(Az[i])
    end
    C
end

function Ac_mul_B!(C::Diagonal{Matrix{T}}, A::FactorReTerm{T}, B::FactorReTerm{T}) where T
    Az = A.wtz
    l, n = size(Az)
    @argcheck A === B && all(d -> size(d) == (l, l), C.diag)
    d = C.diag
    fill!.(d, zero(T))
    refs = A.f.refs
    @inbounds for i in eachindex(refs)
        dri = d[refs[i]]
        for j in 1:l
            Aji = Az[j, i]
            for k in 1:l
                dri[k, j] += Aji * Az[k, i]
            end
        end
    end
    C
end

function Base.Ac_mul_B(A::FactorReTerm{T}, B::FactorReTerm{T}) where T
    if A === B
        l = vsize(A)
        if l == 1
            return Ac_mul_B!(Diagonal(Vector{T}(nlevs(A))), A, A)
        else
            return Ac_mul_B!(Diagonal([zeros(T, (l,l)) for _ in 1:nlevs(A)]), A, A)
        end
    end
    Az = A.wtz
    Bz = B.wtz
    @argcheck(size(Az, 2) == size(Bz, 2), DimensionMismatch)
    m = size(Az, 2)
    a = size(Az, 1)
    b = size(Bz, 1)
    ab = a * b
    nz = ab * m
    I = sizehint!(Int[], nz)
    J = sizehint!(Int[], nz)
    V = sizehint!(T[], nz)
    Ar = A.f.refs
    Br = B.f.refs
    Ipat = repeat(1 : a, outer = b)
    Jpat = repeat(1 : b, inner = a)
    ## A lot of allocation in this block
    for i in 1 : m
        append!(I, Ipat + (Ar[i] - 1) * a)
        append!(J, Jpat + (Br[i] - 1) * b)
        append!(V, vec(view(Az, :, i) * view(Bz, :, i)'))
    end
    sparse(I, J, V)
end

function Ac_mul_B!(C::Matrix{T}, A::FactorReTerm{T}, B::FactorReTerm{T}) where T
    m, n = size(B)
    @argcheck size(C, 1) == size(A, 2) && n == size(C, 2) && size(A, 1) == m DimensionMismatch
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.wtz
    Bz = B.wtz
    Avs = vsize(A)
    Bvs = vsize(B)
    fill!(C, zero(T))
    for i in 1:m, j in 1:Bvs
        jj = (Br[i] - 1) * Bvs + j
        for k in 1:Avs
            C[(Ar[i] - 1)*Avs + k, jj] += Az[k, i] * Bz[j, i]
        end
    end
    C
end

function Ac_mul_B!(C::SparseMatrixCSC{T}, A::FactorReTerm{T}, B::FactorReTerm{T}) where T
    m, n = size(B)
    @argcheck size(C, 1) == size(A, 2) && n == size(C, 2) && size(A, 1) == m DimensionMismatch
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.wtz
    Bz = B.wtz
    Avs = vsize(A)
    Bvs = vsize(B)
    nz = nonzeros(C)
    rv = rowvals(C)
    fill!(nz, zero(T))
    msg1 = "incompatible non-zero pattern in C at row "
    msg2 = " of A and B"
    for i in 1:m, j in 1:Bvs
        nzr = nzrange(C, (Br[i] - 1) * Bvs + j)
        rvalsj = view(rv, nzr)
        irow1 = (Ar[i] - 1) * Avs + 1  # first nonzero row in outer product of row i of A and B
        nzind = searchsortedlast(rvalsj, irow1)
        iszero(nzind) && throw(ArgumentError(string(msg1, i, msg2)))
        inds = view(nzr, nzind - 1 + 1:Avs)
        for k in eachindex(inds)
            rv[inds[k]] == (k + irow1 - 1) || throw(ArgumentError(string(msg1, i, msg2)))
            nz[inds[k]] += Az[k, i] * Bz[j, i]
        end
    end
    C
end
