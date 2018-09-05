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
    piv::Vector{Int}
    rank::Int
    cnames::Vector{String}
end

function MatrixTerm(X::AbstractMatrix, cnms)
    T = eltype(X)
    cf = cholesky!(Symmetric(X'X, :U), Val(true), tol = -one(T))
    r = cf.rank
    piv = cf.piv
    X = X[:, piv[1:r]]
    MatrixTerm{T,typeof(X)}(X, X, piv, r, cnms)
end

function MatrixTerm(y::Vector)
    T = eltype(y)
    m = reshape(y, (length(y), 1))
    MatrixTerm{T,Matrix{T}}(m, m, [1], Int(all(iszero, y)), [""])
end

function reweight!(A::MatrixTerm{T}, sqrtwts::Vector{T}) where T
    if !isempty(sqrtwts)
        if (A.x === A.wtx)
            A.wtx = similar(A.x)
        end
        mul!(A.wtx, Diagonal(sqrtwts), A.x)
    end
    A
end

Base.adjoint(A::AbstractTerm) = Adjoint(A)

Base.eltype(A::MatrixTerm) = eltype(A.wtx)

Base.length(A::MatrixTerm) = length(A.wtx)

Base.size(A::MatrixTerm) = size(A.wtx)

Base.size(A::Adjoint{T,<:MatrixTerm{T}}) where {T} = reverse(size(A.parent))

Base.size(A::MatrixTerm, i) = size(A.wtx, i)

Base.copyto!(A::MatrixTerm{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.x, src)

*(A::Adjoint{T,<:MatrixTerm{T}}, B::MatrixTerm{T}) where {T} = A.parent.wtx'B.wtx

LinearAlgebra.mul!(R::AbstractMatrix{T}, A::MatrixTerm{T}, B::MatrixTerm{T}) where {T} =
    mul!(R, A.wtx, B.wtx)

LinearAlgebra.mul!(R::StridedVecOrMat{T}, A::MatrixTerm{T}, B::StridedVecOrMat{T}) where {T} =
    mul!(R, A.x, B)

LinearAlgebra.mul!(C, A::Adjoint{T,<:MatrixTerm{T}}, B::MatrixTerm{T}) where {T} =
    mul!(C, A.parent.wtx', B.wtx)

abstract type AbstractFactorReTerm{T} <: AbstractTerm{T} end

"""
    isnested(A::AbstractFactorReTerm, B::AbstractFactorReTerm)

Is factor `A` nested in factor `B`?  That is, does each value of `A` occur with just
one value of B?
"""
function isnested(A::AbstractFactorReTerm, B::AbstractFactorReTerm)
    @argcheck length(A.refs) == length(B.refs) DimensionMismatch
    bins = zeros(eltype(B.refs), length(A.levels))
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
* `refs`: indices into `levels` for the grouping factor
* `levels`: possible values of the grouping factor
* `z`: the raw random-effects model matrix as a vector.
* `wtz`: a weighted copy of `z`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnm`: the column name as a string
* `Λ`: the relative covariance multiplier
"""
mutable struct ScalarFactorReTerm{T,R} <: AbstractFactorReTerm{T}
    refs::Vector{R}
    levels::Vector{String}
    z::Vector{T}
    wtz::Vector{T}
    fnm::Symbol
    cnms::Vector{String}
    Λ::T
end
ScalarFactorReTerm(f::CategoricalVector, v::Vector, fnm::Symbol, colnms) =
    ScalarFactorReTerm(CategoricalArrays.order(f.pool)[f.refs], string.(levels(f)), v, copy(v), fnm, colnms, 1.0)
ScalarFactorReTerm(f::CategoricalVector, fnm::Symbol) = ScalarFactorReTerm(f, ones(length(f)), fnm, ["(Intercept)"])
ScalarFactorReTerm(f::AbstractVector, fnm::Symbol) = ScalarFactorReTerm(categorical(f), fnm)

function LinearAlgebra.cond(A::ScalarFactorReTerm)
    Λ = A.Λ
    iszero(Λ) ? oftype(Λ, Inf) : one(Λ)
end

getΛ(A::ScalarFactorReTerm) = A.Λ

function reweight!(A::ScalarFactorReTerm, sqrtwts::Vector)
    if length(sqrtwts) > 0
        if A.z == A.wtz
            A.wtz = A.z .* sqrtwts
        else
            A.wtz .= A.z .* sqrtwts
        end
    end
    A
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

* `refs`: indices into `levels` for the grouping factor
* `levels`: possible values of the grouping factor
* `z`: the transposed raw random-effects model matrix
* `wtz`: a weighted copy of `z`
* `wtzv`: a view of `wtz` as a `Vector{SVector{S,T}}`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
* `blks`: a `Vector{Int}` of block sizes within `Λ`
* `Λ`: the relative covariance factor
* `inds`: linear indices of θ elements in the relative covariance factor
"""
mutable struct VectorFactorReTerm{T,R,S} <: AbstractFactorReTerm{T}
    refs::Vector{R}
    levels::Vector{String}
    z::Matrix{T}
    wtz::Matrix{T}
    wtzv::Base.ReinterpretArray{SVector{S,T}}
    fnm::Symbol
    cnms::Vector{String}
    blks::Vector{Int}
    Λ::LowerTriangular{T,Matrix{T}}
    inds::Vector{Int}
end
function VectorFactorReTerm(f::CategoricalVector, z::Matrix{T}, fnm, cnms, blks) where {T}
    k, n = size(z)
    @argcheck(k == sum(blks), DimensionMismatch)
    m = reshape(1:abs2(k), (k, k))
    offset = 0
    inds = sizehint!(Int[], (k * (k + 1)) >> 1)
    for kk in blks
        for j in 1:kk, i in j:kk
            push!(inds, m[offset + i, offset + j])
        end
        offset += kk
    end
    VectorFactorReTerm(CategoricalArrays.order(f.pool)[f.refs], string.(levels(f)), z, z,
                       reinterpret(SVector{k,T}, vec(z)), fnm, cnms, blks,
                       LowerTriangular(Matrix{T}(I, k, k)), inds)
end

function reweight!(A::VectorFactorReTerm{T,R,S}, sqrtwts::Vector) where {T,R,S}
    if !isempty(sqrtwts)
        z = A.z
        if A.wtz === z
            A.wtz = copy(z)
            A.wtzv = reinterpret(SVector{S,T}, vec(A.wtz))
        end
        A.wtz .= z * Diagonal(sqrtwts)
    end
    A
end

function SparseArrays.sparse(A::ScalarFactorReTerm)
    Az = A.z
    m = length(Az)
    sparse(Vector{Int32}(1:m), Vector{Int32}(A.refs), Az, m, nlevs(A))
end

function SparseArrays.sparse(A::VectorFactorReTerm{T,R,S}) where {T,R,S}
    n = size(A, 1)
    colind = Matrix{Int32}(undef, S, n)
    rr = A.refs
    @inbounds for j in 1:n
        offset = (rr[j] - 1) * S
        for i in 1:S
            colind[i, j] = offset + i
        end
    end
    sparse(Vector{Int32}(repeat(1:n, inner=S)), vec(colind), vec(A.z))
end

"""
    levs(A::AbstractFactorReTerm)

Return the levels of the grouping factor.

# Examples
```jldoctest
julia> trm = ScalarFactorReTerm(categorical(repeat('A':'F', inner = 5)), :G);

julia> show(MixedModels.levs(trm))
['A', 'B', 'C', 'D', 'E', 'F']
```
"""
levs(A::AbstractFactorReTerm) = A.levels

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

Return a vector of the Euclidean lengths of the rows of `A.
Λ`
"""
function rowlengths end

rowlengths(A::ScalarFactorReTerm) = [abs(A.Λ)]

function rowlengths(A::VectorFactorReTerm)
    ld = A.Λ.data
    [norm(view(ld, i, 1:i)) for i in 1:size(ld, 1)]
end

rowlengths(A::MatrixTerm{T}) where {T} = T[]

vsize(A::VectorFactorReTerm{T,R,S}) where {T,R,S} = S

Base.eltype(::AbstractFactorReTerm{T}) where {T} = T

LinearAlgebra.Matrix(A::AbstractFactorReTerm) = Matrix(sparse(A))

Base.size(A::AbstractFactorReTerm) = (length(A.refs), nrandomeff(A))

Base.size(A::AbstractFactorReTerm, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.refs) :  i == 2 ? nrandomeff(A) : 1

LinearAlgebra.cond(A::VectorFactorReTerm) = cond(A.Λ)

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
    m = A.Λ.data
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
function getθ end

getθ(::MatrixTerm{T}) where {T} = T[]
getθ(A::ScalarFactorReTerm) = [A.Λ]
getθ(A::VectorFactorReTerm) = A.Λ.data[A.inds]
getθ(v::Vector{AbstractTerm{T}}) where {T} = vcat(getθ.(v)...)

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
lowerbd(A::VectorFactorReTerm{T}) where {T} = T[x ∈ diagind(A.Λ.data) ? zero(T) : T(-Inf) for x in A.inds]
lowerbd(v::Vector{AbstractTerm{T}}) where {T} = vcat(lowerbd.(v)...)

function setθ!(A::ScalarFactorReTerm{T}, v::T) where {T}
    A.Λ = v
    A
end

function setθ!(A::VectorFactorReTerm{T}, v::AbstractVector{T}) where T
    A.Λ.data[A.inds] = v
    A
end

function αβAc_mul_B!(α::Real, A::MatrixTerm{T}, B::ScalarFactorReTerm{T,R}, β::Real,
                   C::Matrix{T}) where {T,R}
    Awt = A.wtx
    n, p = size(Awt)
    m, q = size(B)
    @argcheck(size(C) == (p, q) && m == n, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : lmul!(β, C)
    end
    rr = B.refs
    zz = B.wtz
    @inbounds for j in 1:n
        rrj = rr[j]
        αzj = α * zz[j]
        for i in 1:p
            C[i, rrj] += αzj * Awt[j, i]
        end
    end
    C
end

LinearAlgebra.mul!(R::Matrix{T}, A::Adjoint{T,<:MatrixTerm{T}}, B::AbstractFactorReTerm{T}) where {T} =
    αβAc_mul_B!(one(T), A.parent, B, zero(T), R)

*(A::Adjoint{T,<:MatrixTerm{T}}, B::AbstractFactorReTerm{T}) where {T} =
    mul!(Matrix{T}(undef, size(A.parent, 2), size(B, 2)), A, B)

function αβAc_mul_B!(α::Real, A::MatrixTerm{T}, B::VectorFactorReTerm{T,R,S}, β::Real,
                   C::Matrix{T}) where {T,R,S}
    Awt = A.wtx
    n, p = size(Awt)
    m, q = size(B)
    @argcheck(size(C) == (p, q) && m == n, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : lmul!(β, C)
    end
    rr = B.refs
    zz = B.wtzv
    @inbounds for j in 1:n
        v = zz[j]
        coloffset = (rr[j] - 1) * S
        for k in 1:S
            jj = coloffset + k
            for i in 1:p
                C[i, jj] += α * Awt[j, i] * v[k]
            end
        end
    end
    C
end

LinearAlgebra.mul!(C::Matrix{T}, A::Adjoint{T,<:MatrixTerm{T}}, B::VectorFactorReTerm{T}) where {T} =
    αβAc_mul_B!(one(T), A.parent, B, zero(T), C)

function LinearAlgebra.mul!(C::Diagonal{T}, A::Adjoint{T,<:ScalarFactorReTerm{T,R}},
                            B::ScalarFactorReTerm{T,R}) where {T,R}
	Ap = A.parent
    @argcheck Ap === B
    d = C.diag
    fill!(d, zero(T))
    @inbounds for (ri, Azi) in zip(Ap.refs, Ap.wtz)
        d[ri] += abs2(Azi)
    end
    C
end

function *(A::Adjoint{T,<:ScalarFactorReTerm{T,R}}, B::ScalarFactorReTerm{T,S}) where {T,R,S}
	Ap = A.parent
    Ap === B ? mul!(Diagonal(Vector{T}(undef, nlevs(B))), A, B) :
        sparse(Vector{Int32}(Ap.refs), Vector{Int32}(B.refs), Ap.wtz .* B.wtz)
end

function *(A::Adjoint{T,<:VectorFactorReTerm{T}}, B::ScalarFactorReTerm{T}) where {T}
	Ap = A.parent
    nzeros = copy(Ap.wtz)
    k, n = size(nzeros)
    rowind = Matrix{Int32}(undef, k, n)
    refs = Ap.refs
    bwtz = B.wtz
    for j in 1:n
        bwtzj = bwtz[j]
        offset = k * (refs[j] - 1)
        for i in 1:k
            rowind[i, j] = i + offset
            nzeros[i, j] *= bwtzj
        end
    end
    sparse(vec(rowind), Vector{Int32}(repeat(B.refs, inner=k)), vec(nzeros),
           k * nlevs(Ap), nlevs(B))
end

*(A::Adjoint{T,<:ScalarFactorReTerm{T}}, B::VectorFactorReTerm{T}) where {T} = adjoint(B'A)

function LinearAlgebra.mul!(C::UniformBlockDiagonal{T}, A::Adjoint{T,<:VectorFactorReTerm{T,R,S}},
                            B::VectorFactorReTerm{T,U,P}) where {T,R,S,U,P}
    @argcheck(A.parent === B)
	Ap = A.parent
    Cd = C.data
    @argcheck(size(Cd) == (S, S, nlevs(B)), DimensionMismatch)
    fill!(Cd, zero(T))
    for (r, v) in zip(Ap.refs, Ap.wtzv)
        @inbounds for j in 1:S
            vj = v[j]
            for i in 1:S
                Cd[i, j, r] += vj * v[i]
            end
        end
    end
    C
end

function *(A::Adjoint{T,<:VectorFactorReTerm{T,R,S}}, B::VectorFactorReTerm{T,U,P}) where {T,R,S,U,P}
    if A.parent === B
        return mul!(UniformBlockDiagonal(Array{T}(undef, S, S, nlevs(B))), A, B)
    end
    Ap = A.parent
    Az = Ap.wtzv
    Bz = B.wtzv
    @argcheck((m = size(Ap, 1)) == size(B, 1), DimensionMismatch)
    ab = S * P
    nz = ab * m
    I = sizehint!(Int32[], nz)
    J = sizehint!(Int32[], nz)
    vals = sizehint!(T[], nz)
    Ar = Ap.refs
    Br = B.refs
    for i in 1:m
        Azi = Az[i]
        Bzi = Bz[i]
        if iszero(Azi) || iszero(Bzi)
            continue
        end
        Ari = Ar[i]
        Bri = Br[i]
        ioffset = (Ari - 1) * S
        joffset = (Bri - 1) * P
        for jj in 1:P
            jjo = jj + joffset
            Bzijj = Bzi[jj]
            for ii in 1:S
                push!(I, ii + ioffset)
                push!(J, jjo)
                push!(vals, Azi[ii] * Bzijj)
            end
        end
    end
    cscmat = sparse(I, J, vals)
    nzs = nonzeros(cscmat)
    q, r = divrem(length(nzs), S)
    iszero(r) || throw(DimensionMismatch("nnz(cscmat) = $(nnz(cscmat)) should be a multiple of $S"))
    nzasmat = reshape(nzs, (S, q))
    rowblocks = [SubArray{T,1,Vector{T}}[] for i in 1:nlevs(Ap)]
    rv = rowvals(cscmat)
    inds = 1:S
    pattern = Vector(inds)
    pattern[S] = 0
    for b in 1:q
        rows = view(rv, inds)
        rows .% S == pattern ||
            throw(ArgumentError("Rows for block $b are not contiguous starting at a multiple of $S"))
        push!(rowblocks[div(rows[1], S) + 1], view(nzs, inds))
        inds = inds .+ S
    end
    nlB = nlevs(B)
    colblocks = sizehint!(StridedMatrix{T}[], nlB)
    colrange = 1:P
    for j in 1:nlB
        inds = nzrange(cscmat, colrange[1])
        rows = rv[inds]
        i1 = inds[1]
        for k in 2:P
            inds = nzrange(cscmat, colrange[k])
            rv[inds] == rows || throw(DimensionMismatch("Rows differ ($rows ≠ $(rv[inds])) at column block $j"))
        end
        push!(colblocks, reshape(view(nzs, i1:inds[end]), (length(rows), P)))
        colrange = colrange .+ P
    end
    BlockedSparse(cscmat, nzasmat, rowblocks, colblocks)
end

function LinearAlgebra.mul!(C::Matrix{T}, A::Adjoint{T,<:ScalarFactorReTerm{T}}, B::ScalarFactorReTerm{T}) where T
    m, n = size(B)
	Ap = A.parent
    @argcheck size(C, 1) == size(Ap, 2) && n == size(C, 2) && size(Ap, 1) == m DimensionMismatch
    Ar = Ap.refs
    Br = B.refs
    Az = Ap.wtz
    Bz = B.wtz
    fill!(C, zero(T))
    for i in 1:m
        C[Ar[i], Br[i]] += Az[i] * Bz[i]
    end
    C
end

function LinearAlgebra.mul!(C::SparseMatrixCSC{T}, A::Adjoint{T,<:ScalarFactorReTerm{T}}, B::ScalarFactorReTerm{T}) where T
    m, n = size(B)
	Ap = A.parent
    @argcheck size(C, 1) == size(Ap, 2) && n == size(C, 2) && size(Ap, 1) == m DimensionMismatch
    Ar = Ap.refs
    Br = B.refs
    Az = Ap.wtz
    Bz = B.wtz
    nz = nonzeros(C)
    rv = rowvals(C)
    fill!(nz, zero(T))
    for i in 1:m
        rng = nzrange(C, Br[i])
        k = findfirst(view(rv, rng), Ar[i])
        iszero(k) && throw(ArgumentError("C is not compatible with A and B at index $i"))
        nz[rng[k]] += Az[i] * Bz[i]
    end
    C
end

function LinearAlgebra.mul!(C::Matrix{T}, A::Adjoint{T,<:VectorFactorReTerm{T,R,S}},
                            B::ScalarFactorReTerm{T}) where {T,R,S}
    m, n = size(B)
	Ap = A.parent
    @argcheck size(C, 1) == size(Ap, 2) && n == size(C, 2) && size(Ap, 1) == m DimensionMismatch
    Ar = Ap.refs
    Br = B.refs
    Az = Ap.wtz
    Bz = B.wtz
    fill!(C, zero(T))
    for j in 1:m
        offset = S * (Ar[j] - 1)
        Bzj = Bz[j]
        Brj = Br[j]
        for k in 1:S
            C[offset + k, Brj] += Az[k,j] * Bzj
        end
    end
    C
end

function LinearAlgebra.mul!(C::Matrix{T}, A::Adjoint{T,<:ScalarFactorReTerm{T}},
	                        B::VectorFactorReTerm{T,R,S}) where {T,R,S}
    m, n = size(B)
	Ap = A.parent
    @argcheck size(C, 1) == size(Ap, 2) && n == size(C, 2) && size(Ap, 1) == m DimensionMismatch
    Ar = Ap.refs
    Br = B.refs
    Az = Ap.wtz
    Bz = B.wtz
    fill!(C, zero(T))
    for j in 1:m
        offset = S * (Br[j] - 1)
        Azj = Az[j]
        Arj = Ar[j]
        for k in 1:S
            C[Arj, offset + k] += Azj * Bz[k,j]
        end
    end
    C
end
