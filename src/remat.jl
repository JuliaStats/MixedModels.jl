
@compat const AbstractFactor{V,R} = Union{NullableCategoricalVector{V,R},CategoricalVector{V,R},PooledDataVector{V,R}}

"""
    asfactor(f)

Return `f` as a AbstractFactor.

This function and the `AbstractFactor` union can be removed once `CategoricalArrays` replace
`PooledDataArray`
"""
asfactor(f::AbstractFactor) = f
asfactor(f) = pool(f)

"""
    ReMat

Representation of the model matrix for random-effects terms

# Members
* `f`: the grouping factor as an `AbstractFactor`
* `z`: the transposed raw random-effects model matrix
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
"""
immutable ReMat{T<:AbstractFloat,V,R}
    f::AbstractFactor{V,R}
    z::Matrix{T}
    fnm::Symbol
    cnms::Vector
end

"""
    levs(A::ReMat)

Return the levels of the grouping factor.

This is to disambiguate a call to `levels` as both `DataArrays`
and `CategoricalArrays` export it.
"""
function levs(A::ReMat)
    f = A.f
    isa(f, PooledDataArray) ? DataArrays.levels(f) : CategoricalArrays.levels(f)
end

"""
    nlevs(A::ReMat)

Return the number of levels in the grouping factor of `A`.
"""
nlevs(A::ReMat) = length(levs(A))

"""
    nrandomeff(A::ReMat)

Return the total number of random effects in A.
"""
nrandomeff(A::ReMat) = nlevs(A) * size(A.z, 1)

Base.eltype(R::ReMat) = eltype(R.z)

Base.full(R::ReMat) = full(sparse(R))

"""
    vsize(A::ReMat)

Return the size of vector-valued random effects.
"""
vsize(A::ReMat) = size(A.z, 1)

Base.size(A::ReMat) = (length(A.f), vsize(A) * nlevs(A))

Base.size(A::ReMat, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.f) :  i == 2 ? vsize(A)*nlevs(A) : 1

Base.sparse(R::ReMat) =
    sparse(Int32[1:length(R.z);], Int32[repeat(R.f.refs, inner=size(R.z, 1))], vec(R.z))

==(A::ReMat,B::ReMat) = (A.f == B.f) && (A.z == B.z)

function A_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n,q = size(A)
    k = size(B, 2)
    @argcheck size(R, 1) == n && size(B, 1) == q && size(R, 2) == k DimensionMismatch
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = A.f.refs
    zz = A.z
    if vsize(A) == 1
        for j in 1:k, i in 1:n
            R[i, j] += α * zz[i] * B[rr[i],j]
        end
    else
        l = size(zz, 1)
        Bt = reshape(B, (l, div(q, l), k))
        for j in 1:k, i in 1:n
            R[i, j] += α * dot(view(Bt, :, Int(rr[i]), j), view(zz, :, i))
        end
    end
    R
end

A_mul_B!{T}(A::ReMat, B::StridedVecOrMat{T}, R::StridedVecOrMat{T}) = A_mul_B!(one(T), A, B, zero(T), R)

function Ac_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n, q = size(A)
    k = size(B, 2)
    @argcheck size(R, 1) == q && size(B, 1) == n && size(R, 2) == k DimensionMismatch
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = A.f.refs
    zz = A.z
    if vsize(A) == 1
        for j in 1:k, i in 1:n
            R[rr[i], j] += α * zz[i] * B[i, j]
        end
    else
        l = size(zz, 1)
        for j in 1 : k, i in 1 : n
            roffset = (rr[i] - 1) * l
            mul = α * B[i, j]
            for ii in 1 : l
                R[roffset + ii, j] += mul * zz[ii, i]
            end
        end
    end
    R
end

Ac_mul_B!{T}(R::StridedVecOrMat{T}, A::ReMat, B::StridedVecOrMat{T}) =
    Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::ReMat, B::DenseVecOrMat)
    k = size(A, 2)
    Ac_mul_B!(zeros(eltype(B), isa(B, Vector) ? (k,) : (k, size(B, 2))), A, B)
end

function Ac_mul_B!{T}(C::Diagonal{Matrix{T}}, A::ReMat{T}, B::ReMat{T})
    A === B || throw(ArgumentError("method only makes sense for A === B"))
    Az = A.z
    l, n = size(Az)
    d = C.diag
    fill!.(d, zero(T))
    all(size.(d, 2) .== l) || throw(ArgumentError("A and C do not conform"))
    refs = A.f.refs
    for i in eachindex(refs)
        rankUpdate!(view(Az, :, i), Hermitian(d[refs[i]], :L))
    end
    map!(m -> copytri!(m, 'L'), d, d)
    C
end

function Base.Ac_mul_B{T}(A::ReMat{T}, B::ReMat{T})
    if A === B
        l = vsize(A)
        return Ac_mul_B!(Diagonal([zeros(T, (l,l)) for _ in 1:nlevs(A)]), A, A)
    end
    Az = A.z
    Bz = B.z
    @argcheck size(Az, 2) == size(Bz, 2) DimensionMismatch
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
    for i in 1 : m
        append!(I, Ipat + (Ar[i] - 1) * a)
        append!(J, Jpat + (Br[i] - 1) * b)
        append!(V, vec(view(Az, :, i) * view(Bz, :, i)'))
    end
    sparse(I, J, V)
end

function Ac_mul_B!{T}(R::DenseVecOrMat{T}, A::DenseVecOrMat{T}, B::ReMat)
    m = size(A, 1)
    n = size(A, 2)  # needs to be done this way in case A is a vector
    p, q = size(B)
    @argcheck m == p && size(R, 1) == n && size(R, 2) == q DimensionMismatch
    fill!(R, 0)
    r, z = B.f.refs, B.z
    if vsize(B) == 1
        for j in 1:n, i in 1:m
            R[j, r[i]] += A[i, j] * z[i]
        end
    else
        l = size(z, 1)
        for j in 1:n, i in 1:m
            roffset = (r[i] - 1) * l
            aij = A[i, j]
            for k in 1:l
                R[j, roffset + k] += aij * z[k, i]
            end
        end
    end
    R
end

function Ac_mul_B!{T}(C::Matrix{T}, A::ReMat{T}, B::ReMat{T})
    m, n = size(B)
    @argcheck size(C, 1) == size(A, 2) && n == size(C, 2) && size(A, 1) == m DimensionMismatch
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.z
    Bz = B.z
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

function Ac_mul_B!{T}(C::SparseMatrixCSC{T}, A::ReMat{T}, B::ReMat{T})
    m, n = size(B)
    @argcheck size(C, 1) == size(A, 2) && n == size(C, 2) && size(A, 1) == m DimensionMismatch
    Ar = A.f.refs
    Br = B.f.refs
    Az = A.z
    Bz = B.z
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

Base.Ac_mul_B(A::DenseVecOrMat, B::ReMat) = Ac_mul_B!(Array{eltype(A)}((size(A, 2), size(B, 2))), A, B)

function A_mul_B!{T}(A::Diagonal{T}, B::ReMat{T})
    scale!(B.z, A.diag)
    B
end

(*){T}(D::Diagonal{T}, A::ReMat{T}) = ReMat(A.f, A.z * D, A.fnm, A.cnms)

function A_mul_B!{T}(C::ReMat{T}, A::Diagonal{T}, B::ReMat{T})
    A_mul_B!(C.z, B.z, A)
    C
end
