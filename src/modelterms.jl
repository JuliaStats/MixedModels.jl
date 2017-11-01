"""
    MatrixTerm

Term with an explicit, constant matrix representation

# Members
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

abstract type AbstractFactorReTerm{T} <: AbstractTerm{T} end

"""
    isnested(A, B)

Is factor `A` nested in factor `B`?  That is, does each value of `A` occur with just
one value of B?
"""
function isnested(A::PooledDataVector{V,R}, B::PooledDataVector{W,S}) where {V,R,W,S}
    @argcheck length(A) == length(B) DimensionMismatch
    bins = zeros(S, length(A.pool))
    @inbounds for (a, b) in zip(A.refs, B.refs)
        bba = bins[a]
        if iszero(bba)
            bins[a] = b
        elseif bba ≠ b
            return false
        end
    end
    true
end

"""
    ScalarFactorReTerm

Scalar random-effects term from a grouping factor

# Members
* `f`: the grouping factor as a `PooledDataArray` (soon to switch to CategoricalArray)
* `z`: the raw random-effects model matrix as a vector.
* `wtz`: a weighted copy of `z`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnm`: the column name as a string
* `Λ`: the relative covariance multiplier
"""
mutable struct ScalarFactorReTerm{T,V,R} <: AbstractFactorReTerm{T}
    f::PooledDataVector{V,R}
    z::Vector{T}
    wtz::Vector{T}
    fnm::Symbol
    cnms::Vector{String}
    Λ::T
end
function ScalarFactorReTerm(f::PooledDataVector, fnm::Symbol)
    v = ones(Float64, length(f))
    ScalarFactorReTerm(f, v, v, fnm, ["(Intercept)"], 1.0)
end
ScalarFactorReTerm(f::AbstractVector, fnm::Symbol) = ScalarFactorReTerm(pool(f), fnm)

function cond(A::ScalarFactorReTerm)
    Λ = A.Λ
    iszero(Λ) ? oftype(Λ, Inf) : one(Λ)
end

getΛ(A::ScalarFactorReTerm) = A.Λ

function reweight!(A::ScalarFactorReTerm, sqrtwts::Vector)
    n = length(sqrtwts)
    if n > 0
        if A.z == A.wtz
            A.wtz = A.z .* sqrtwts
        else
            A.wtz .= A.z .* sqrtwts
        end
    end
    A
end

function Base.sparse(A::ScalarFactorReTerm{T,V,R}) where {T,V,R}
    Af = A.f
    Az = A.z
    m = length(Az)
    sparse(collect(one(Int32):Int32(m)), Af.refs, Az, m, nlevs(A))
end

"""
vsize(A::AbstractFactorReTerm)

Return the size of vector-valued random effects.
"""
function vsize end

vsize(A::ScalarFactorReTerm) = 1


"""
    VectorFactorReTerm

Random-effects term from a grouping factor, model matrix and block pattern

# Members
* `f`: the grouping factor as an `PooledDataVector`
* `z`: the transposed raw random-effects model matrix
* `wtz`: a weighted copy of `z`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
* `blks`: a `Vector{Int}` of block sizes within `Λ`
* `Λ`: the relative covariance factor
* `inds`: linear indices of θ elements in the relative covariance factor
"""
mutable struct VectorFactorReTerm{T,V,R} <: AbstractFactorReTerm{T}
    f::PooledDataVector{V,R}
    z::Matrix{T}
    wtz::Matrix{T}
    fnm::Symbol
    cnms::Vector{String}
    blks::Vector{Int}
    Λ::Matrix{T}
    inds::Vector{Int}
end
function VectorFactorReTerm(f::PooledDataVector, z::Matrix, fnm, cnms, blks)
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
    VectorFactorReTerm(f, z, z, fnm, cnms, blks, eye(eltype(z), n), inds)
end

function reweight!(A::VectorFactorReTerm, sqrtwts::Vector)
    if !isempty(sqrtwts)
        if A.z === A.wtz
            A.wtz = similar(A.z)
        end
        scale!(A.wtz, A.z, sqrtwts)
    end
    A
end

function Base.sparse(A::VectorFactorReTerm{T,V,R}) where {T,V,R}
    l, n = size(A.z)
    I = repeat(one(Int32):Int32(n), inner=l)
    Ar = A.f.refs
    J = vec([Int32((Ar[j] - 1) * l + i) for i in 1:l, j in 1:n])
    sparse(I, J, vec(A.z), n, nrandomeff(A))
end

"""
    levs(A::AbstractFactorReTerm)

Return the levels of the grouping factor.

# Examples
```jldoctest
julia> trm = FactorReTerm(pool(repeat('A':'F', inner = 5)));

julia> show(MixedModels.levs(trm))
['A', 'B', 'C', 'D', 'E', 'F']
julia>
```
"""
levs(A::AbstractFactorReTerm) = levels(A.f)

"""
    nlevs(A::FactorReTerm)

Return the number of levels in the grouping factor of `A`.
"""
nlevs(A::AbstractFactorReTerm) = length(levs(A))

"""
    nrandomeff(A::FactorReTerm)

Return the total number of random effects in A.
"""
nrandomeff(A::AbstractFactorReTerm) = nlevs(A) * vsize(A)

"""
    rowlengths(A::AbstractTerm)

Return a vector of the row lengths of the `Λ`
"""
function rowlengths end

rowlengths(A::ScalarFactorReTerm) = [abs(A.Λ)]

function rowlengths(A::VectorFactorReTerm)
    ld = A.Λ
    [norm(view(ld, i, 1:i)) for i in 1:size(ld, 1)]
end

rowlengths(A::MatrixTerm{T}) where {T} = T[]

vsize(A::VectorFactorReTerm) = size(A.z, 1)

Base.eltype(::AbstractFactorReTerm{T}) where {T} = T

Base.full(R::AbstractFactorReTerm) = full(sparse(R))

Base.size(A::AbstractFactorReTerm) = (length(A.f), nrandomeff(A))

Base.size(A::AbstractFactorReTerm, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.f) :  i == 2 ? nrandomeff(A) : 1

cond(A::VectorFactorReTerm) = cond(LowerTriangular(A.Λ))

"""
    nθ(A::AbstractTerm)

Return the number of free parameters in the relative covariance matrix Λ
"""
function nθ end

nθ(::MatrixTerm) = 0
nθ(::ScalarFactorReTerm) = 1
nθ(A::VectorFactorReTerm) = length(A.inds)

getΛ(A::VectorFactorReTerm) = A.Λ

"""
    getθ!{T}(v::AbstractVector{T}, A::FactorReTerm{T})

Overwrite `v` with the elements of the blocks in the lower triangle of `A.Λ` (column-major ordering)
"""
function getθ! end

function getθ!(v::StridedVector, A::ScalarFactorReTerm)
    @argcheck(length(v) == 1, DimensionMismatch)
    v[1] = A.Λ
    v
end

function getθ!(v::StridedVector{T}, A::VectorFactorReTerm{T}) where T
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
    getθ(A::AbstractFactorReTerm)

Return a vector of the elements of the lower triangle blocks in `A.Λ` (column-major ordering)
"""
function getθ(A::AbstractFactorReTerm) end

getθ(::MatrixTerm{T}) where {T} = T[]
getθ(A::ScalarFactorReTerm) = [A.Λ]
getθ(A::VectorFactorReTerm) = A.Λ[A.inds]
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

lowerbd(::MatrixTerm{T}) where {T} = T[]
lowerbd(A::ScalarFactorReTerm{T}) where {T} = zeros(T, 1)
lowerbd(A::VectorFactorReTerm{T}) where {T} =
    T[x ∈ diagind(A.Λ) ? zero(T) : convert(T, -Inf) for x in A.inds]
lowerbd(v::Vector{AbstractTerm{T}}) where {T} = reduce(append!, T[], lowerbd(t) for t in v)

function setθ!(A::ScalarFactorReTerm{T}, v::T) where {T}
    A.Λ = v
    A
end

function setθ!(A::VectorFactorReTerm{T}, v::AbstractVector{T}) where T
    @argcheck(length(v) == length(A.inds), DimensionMismatch)
    m = A.Λ
    inds = A.inds
    @inbounds for i in eachindex(inds)
        m[inds[i]] = v[i]
    end
    A
end

function Ac_mul_B!(α::Real, A::VectorFactorReTerm{T,V,R}, B::MatrixTerm{T}, β::Real,
                   C::Matrix{T}) where {T,V,R}
    n, q = size(A)
    Bwt = B.wtx
    k = size(Bwt, 2)
    @argcheck(size(C, 1) == q && size(Bwt, 1) == n && size(C, 2) == k, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : scale!(β, C)
    end
    rr = A.f.refs
    Awtz = A.wtz
    l = vsize(A)
    for j in 1:k, i in 1:n
        roffset = (rr[i] - 1) * l
        mul = α * Bwt[i, j]
        for ii in 1 : l
            C[roffset + ii, j] += mul * Awtz[ii, i]
        end
    end
    C
end

function Ac_mul_B!(α::Real, A::ScalarFactorReTerm{T,V,R}, B::MatrixTerm{T}, β::Real,
    C::Matrix{T}) where {T,V,R}
    n, q = size(A)
    Bwt = B.wtx
    k = size(Bwt, 2)
    @argcheck(size(C, 1) == q && size(Bwt, 1) == n && size(C, 2) == k, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : scale!(β, C)
    end
    rr = A.f.refs
    Awtz = A.wtz
    for j in 1:k, i in 1:n
        C[rr[i], j] += α * Awtz[i] * Bwt[i, j]
    end
    C
end

Ac_mul_B!(R::Matrix{T}, A::AbstractFactorReTerm{T}, B::MatrixTerm{T}) where {T} =
    Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::AbstractFactorReTerm{T}, B::MatrixTerm{T}) where T
    Ac_mul_B!(zeros(eltype(B), (size(A, 2), size(B, 2))), A, B)
end

function Ac_mul_B!(α::Real, A::MatrixTerm{T}, B::ScalarFactorReTerm{T,V,R}, β::Real,
                   C::Matrix{T}) where {T,V,R}
    n, p = size(A)
    q = size(B, 2)
    @argcheck(size(C, 1) == p && size(B, 1) == n && size(C, 2) == q, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : scale!(β, C)
    end
    rr = B.f.refs
    zz = B.wtz
    Awt = A.wtx
    @inbounds for j in 1:n
        rrj = rr[j]
        αzj = α * zz[j]
        for i in 1:p
            C[i, rrj] += αzj * Awt[j, i]
        end
    end
    C
end

function Ac_mul_B!(α::Real, A::MatrixTerm{T}, B::VectorFactorReTerm{T,V,R}, β::Real,
                   C::Matrix{T}) where {T,V,R}
    Awt = A.wtx
    n, p = size(Awt)
    q = size(B, 2)
    @argcheck(size(C, 1) == p && size(B, 1) == n && size(C, 2) == q, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : scale!(β, C)
    end
    rr = B.f.refs
    zz = B.wtz
    l = size(zz, 1)
    for j in 1:p, i in 1:n
        roffset = (rr[i] - 1) * l
        mul = α * Awt[i, j]
        for ii in 1:l
            C[j, roffset + ii] += mul * zz[ii, i]
        end
    end
    C
end

Ac_mul_B!(R::Matrix{T}, A::MatrixTerm{T}, B::AbstractFactorReTerm{T}) where {T} =
    Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::MatrixTerm{T}, B::AbstractFactorReTerm{T}) where T
    Ac_mul_B!(Matrix{T}(size(A, 2), size(B, 2)), A, B)
end

function Ac_mul_B!(C::Diagonal{T}, A::ScalarFactorReTerm{T,V,R}, 
                   B::ScalarFactorReTerm{T,V,R}) where {T,V,R}
    @argcheck A === B
    d = C.diag
    fill!(d, zero(T))
    @inbounds for (ri, Azi) in zip(A.f.refs, A.wtz)
        d[ri] += abs2(Azi)
    end
    C
end

function Ac_mul_B!(C::UniformBlockDiagonal{T}, A::VectorFactorReTerm{T,V,R},
                   B::VectorFactorReTerm{T,V,R}) where {T,V,R}
    Az = A.wtz
    l, n = size(Az)
    fv = C.facevec
    @argcheck A === B && size(fv[1]) == (l, l)
    fill!(C.data, zero(T))
    refs = A.f.refs
    @inbounds for i in 1:n
        BLAS.syr!('L', one(T), view(Az, :, i), fv[refs[i]])
    end
    copytri!.(fv, 'L')
    C
end

function Base.Ac_mul_B(A::ScalarFactorReTerm{T,V,R}, 
                       B::ScalarFactorReTerm{T,W,S}) where {T,V,R,W,S}
    if A == B 
        Ac_mul_B!(Diagonal(Vector{T}(nlevs(A))), A, B)
    else
        sparse(Vector{Int32}(A.f.refs), Vector{Int32}(B.f.refs), A.wtz .* B.wtz,
               nlevs(A), nlevs(B))
    end
end

function Base.Ac_mul_B(A::VectorFactorReTerm, B::ScalarFactorReTerm)
    nzeros = copy(A.wtz)
    k, n = size(nzeros)
    rowind = Matrix{Int32}(k, n)
    refs = A.f.refs
    bwtz = B.wtz
    for j in 1:n
        bwtzj = bwtz[j]
        offset = k * (refs[j] - 1)
        for i in 1:k
            rowind[i, j] = i + offset
            nzeros[i, j] *= bwtzj
        end
    end
    sparse(vec(rowind), Vector{Int32}(repeat(B.f.refs, inner=k)), vec(nzeros),
           k * nlevs(A), nlevs(B))
end

Base.Ac_mul_B(A::ScalarFactorReTerm, B::VectorFactorReTerm) = Ac_mul_B(B, A)'

function Base.Ac_mul_B(A::VectorFactorReTerm{T}, B::VectorFactorReTerm{T}) where T
    if A === B
        l = vsize(A)
        nl = nlevs(A)
        return Ac_mul_B!(UniformBlockDiagonal([Matrix{T}(l, l) for _ in 1:nl]), A, A)
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

function Ac_mul_B!(C::Matrix{T}, A::ScalarFactorReTerm{T}, B::ScalarFactorReTerm{T}) where T
    m, n = size(B)
    @argcheck size(C, 1) == size(A, 2) && n == size(C, 2) && size(A, 1) == m DimensionMismatch
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.wtz
    Bz = B.wtz
    fill!(C, zero(T))
    for i in 1:m
        C[Ar[i], Br[i]] += Az[i] * Bz[i]
    end
    C
end

#=  Methods not tested.  Comment out to see if they are really needed.
function Ac_mul_B!(C::Matrix{T}, A::VectorFactorReTerm{T}, B::VectorFactorReTerm{T}) where T
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

function Ac_mul_B!(C::SparseMatrixCSC{T}, A::ScalarFactorReTerm{T},
                   B::ScalarFactorReTerm{T}) where T
    m, n = size(B)
    @argcheck(size(C,1) == size(A,2) && n == size(C,2) && size(A,1) == m, DimensionMismatch)
    nz = nonzeros(C)
    fill!(nz, 0)
    rv = rowvals(C)
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.wtz
    Bz = B.wtz
    for i in 1:m
        nzBr = nzrange(C, Br[i])
        error("Code not yet written")
    end
    C
end


function Ac_mul_B!(C::SparseMatrixCSC{T}, A::VectorFactorReTerm{T}, B::VectorFactorReTerm{T}) where T
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
=#
