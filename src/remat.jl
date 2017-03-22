"""
    ReMat

A representation of the model matrix for a random-effects term
"""
abstract type ReMat end

"""
    ScalarReMat

The representation of the model matrix for a scalar random-effects term

# Members
* `f`: the grouping factor as a `CategoricalVector`
* `z`: the raw random-effects model matrix as a `Vector`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names
"""
struct ScalarReMat{T<:AbstractFloat,V,R} <: ReMat
    f::Union{NullableCategoricalVector{V,R},CategoricalVector{V,R},PooledDataVector{V,R}}
    z::Vector{T}
    fnm::Symbol
    cnms::Vector{String}
end

"""
    VectorReMat

The representation of the model matrix for a vector-valued random-effects term

# Members
* `f`: the grouping factor as a `CategoricalVector`
* `z`: the transposed raw random-effects model matrix
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
"""
struct VectorReMat{T<:AbstractFloat,V,R} <: ReMat
    f::Union{NullableCategoricalVector{V,R},CategoricalVector{V,R},PooledDataVector{V,R}}
    z::Matrix{T}
    fnm::Symbol
    cnms::Vector
end

"""
     remat(e::Expr, df::DataFrames.DataFrame)

A factory for `ReMat` objects.

`e` should be of the form `:(e1 | e2)` where `e1` is a valid rhs of a `Formula` and
`pool(e2)` can be evaluated within `df`.  The result is a
[`ScalarReMat`](@ref) or a [`VectorReMat`](@ref), as appropriate.
"""
function remat(e::Expr, df::DataFrame)
    isa(e, Expr) && e.args[1] == :| || throw(ArgumentError("$e is not a call to '|'"))
    fnm = e.args[3]
    gr = getindex(df, fnm)
    gr = isa(gr, Union{NullableCategoricalVector,CategoricalVector,PooledDataVector}) ? gr : pool(gr)
    if e.args[2] == 1
        return ScalarReMat(gr, ones(length(gr)), fnm, ["(Intercept)"])
    end
    mf = ModelFrame(Formula(nothing, e.args[2]), df)
    z = ModelMatrix(mf).m
    cnms = coefnames(mf)
    size(z,2) == 1 ? ScalarReMat(gr, vec(z), fnm, cnms) : VectorReMat(gr, z', fnm, cnms)
end

Base.eltype(R::ReMat) = eltype(R.z)

Base.full(R::ScalarReMat) = full(sparse(R))

"""
    vsize(A::ReMat)

Return the size of vector-valued random effects (i.e. 1 for a [`ScalarReMat`](@ref)).
"""
vsize(A::ReMat) = isa(A,ScalarReMat) ? 1 : size(A.z, 1)

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
nrandomeff(A::ScalarReMat) = nlevs(A)
nrandomeff(A::VectorReMat) = nlevs(A) * size(A.z, 1)

Base.size(A::ReMat) = (length(A.f), vsize(A) * nlevs(A))

Base.size(A::ReMat, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.f) :  i == 2 ? vsize(A)*nlevs(A) : 1

Base.sparse(R::ScalarReMat) =
    sparse(convert(Vector{Int32}, 1:length(R.z)), convert(Vector{Int32}, R.f.refs), R.z)

==(A::ReMat,B::ReMat) = (A.f == B.f) && (A.z == B.z)

function Base.A_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n,q = size(A)
    k = size(B, 2)
    @argcheck size(R, 1) == n && size(B, 1) == q && size(R, 2) == k DimensionMismatch
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = A.f.refs
    zz = A.z
    if isa(A, ScalarReMat)
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

Base.A_mul_B!{T}(A::ReMat, B::StridedVecOrMat{T}, R::StridedVecOrMat{T}) = A_mul_B!(one(T), A, B, zero(T), R)

function Ac_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n, q = size(A)
    k = size(B, 2)
    @argcheck size(R, 1) == q && size(B, 1) == n && size(R, 2) == k DimensionMismatch
    if β ≠ one(T)
        iszero(β) ? fill!(R, β) : scale!(β, R)
    end
    rr = A.f.refs
    zz = A.z
    if isa(A, ScalarReMat)
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

function Base.Ac_mul_B{T}(A::ScalarReMat{T}, B::ScalarReMat{T})
    Az, Ar = A.z, A.f.refs
    if A === B
        v = zeros(T, nlevs(A))
        for i in eachindex(Ar)
            v[Ar[i]] += abs2(Az[i])
        end
        return Diagonal(v)
    end
    densify(sparse(convert(Vector{Int32}, Ar), convert(Vector{Int32}, B.f.refs), Az .* B.z))
end

function Base.Ac_mul_B{T}(A::VectorReMat{T}, B::ScalarReMat{T})
    @argcheck size(A, 1) == size(B, 1) DimensionMismatch
    k = Int32(vsize(A))
    seq = one(Int32) : k
    rowvals = sizehint!(Int32[], size(A, 2))
    for j in A.f.refs
        append!(rowvals, seq + k * (j - one(Int32)))
    end
    densify(sparse(rowvals, convert(Vector{Int32}, repeat(B.f.refs, inner = k)),
        vec(A.z * Diagonal(B.z))))
end

Base.Ac_mul_B{T}(A::ScalarReMat{T}, B::VectorReMat{T}) = ctranspose(B'A)

function Ac_mul_B!{T}(C::Diagonal{T}, A::ScalarReMat{T}, B::ScalarReMat{T})
    c, a, r, b = C.diag, A.z, A.f.refs, B.z
    if r ≠ B.f.refs
        throw(ArgumentError("A'B is not diagonal"))
    end
    fill!(c, 0)
    for i in eachindex(a)
        c[r[i]] += a[i] * b[i]
    end
    C
end

function Ac_mul_B!{Tv, Ti}(C::SparseMatrixCSC{Tv, Ti}, A::ScalarReMat{Tv}, B::ScalarReMat{Tv})
    m, n = size(A)
    p, q = size(B)
    @argcheck size(A, 1) == size(B, 1) && size(C, 1) == size(A, 2) && size(C, 2) == size(B, 2) DimensionMismatch
    SparseArrays.sparse!(convert(Vector{Ti}, A.f.refs), convert(Vector{Ti}, B.f.refs), A.z .* B.z,
        n, q, +, Array{Ti}(q), Array{Ti}(n + 1), Array{Ti}(m), Array{Tv}(m), C.colptr, C.rowval, C.nzval)
    C
end

function Ac_mul_B!{T}(C::Matrix{T}, A::ScalarReMat{T}, B::ScalarReMat{T})
    m, n = size(C)
    ma, na = size(A)
    mb, nb = size(B)
    @argcheck m == na && n == nb && ma == mb DimensionMismatch
    a = A.z
    b = B.z
    ra = A.f.refs
    rb = B.f.refs
    fill!(C, 0)
    for i in eachindex(a)
        C[ra[i], rb[i]] += a[i] * b[i]
    end
    C
end

function Base.Ac_mul_B{T}(A::VectorReMat{T}, B::VectorReMat{T})
    if A === B
        Az = A.z
        refs = A.f.refs
        l, n = size(Az)
        D = Diagonal([zeros(T, (l,l)) for _ in 1:nlevs(A)])
        d = D.diag
        for i in eachindex(refs)
            rankUpdate!(view(Az, :, i), Hermitian(d[refs[i]], :L))
        end
        map!(m -> copytri!(m, 'L'), d, d)
        return D
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
    if isa(B, ScalarReMat)
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

Base.Ac_mul_B(A::DenseVecOrMat, B::ReMat) = Ac_mul_B!(Array{eltype(A)}((size(A, 2), size(B, 2))), A, B)

(*){T}(D::Diagonal{T}, A::ScalarReMat{T}) = ScalarReMat(A.f, D * A.z, A.fnm, A.cnms)

function Base.A_mul_B!{T}(C::ScalarReMat{T}, A::Diagonal{T}, B::ScalarReMat{T})
    map!(*, C.z, A.diag, B.z)
    C
end

function Base.A_mul_B!{T}(A::Diagonal{T}, B::ScalarReMat{T})
    B.z .*= A.diag
    B
end

function Base.A_mul_B!{T}(A::Diagonal{T}, B::VectorReMat{T})
    scale!(B.z, A.diag)
    B
end

(*){T}(D::Diagonal{T}, A::VectorReMat{T}) = VectorReMat(A.f, A.z * D, A.fnm, A.cnms)

function Base.A_mul_B!{T}(C::VectorReMat{T}, A::Diagonal{T}, B::VectorReMat{T})
    A_mul_B!(C.z, B.z, A)
    C
end
