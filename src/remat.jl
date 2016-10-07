"""
    ReMat

A representation of the model matrix for a random-effects term
"""
abstract ReMat

"""
    ScalarReMat

The representation of the model matrix for a scalar random-effects term

# Members
* `f`: the grouping factor as a `PooledDataVector`
* `z`: the raw random-effects model matrix as a `Vector`
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names
"""
immutable ScalarReMat{T <: AbstractFloat, S, R <: Integer} <: ReMat
    f::PooledDataVector{S,R}
    z::Vector{T}
    fnm::Symbol
    cnms::Vector
end

"""
    VectorReMat

The representation of the model matrix for a vector-valued random-effects term

# Members
* `f`: the grouping factor as a `PooledDataVector`
* `z`: the transposed raw random-effects model matrix
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
"""
immutable VectorReMat{T <: AbstractFloat, S, R <: Integer} <: ReMat
    f::PooledDataVector{S,R}
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
    e.args[1] == :| || throw(ArgumentError("$e is not a call to '|'"))
    fnm = e.args[3]
    gr = getindex(df, fnm)
    gr = isa(gr,PooledDataArray) ? gr : pool(gr)
    if e.args[2] == 1
        return ScalarReMat(gr, ones(length(gr)), fnm, ["(Intercept)"])
    end
    mf = ModelFrame(Formula(nothing, e.args[2]), df)
    z = ModelMatrix(mf).m
    cnms = coefnames(mf)
    size(z,2) == 1 ? ScalarReMat(gr, vec(z), fnm, cnms) : VectorReMat(gr, z', fnm, cnms)
end

Base.eltype(R::ReMat) = eltype(R.z)

function Base.copy!{S,R}(d::PooledDataVector{S,R}, s::PooledDataVector{S,R})
    copy!(d.pool, s.pool)
    copy!(d.refs, s.refs)
    d
end

function Base.copy!{T,S,R}(d::ScalarReMat{T,S,R}, s::ScalarReMat{T,S,R})
    copy!(d.f, s.f)
    copy!(d.z, s.z)
    d
end

function Base.copy!{T,S,R}(d::VectorReMat{T,S,R}, s::VectorReMat{T,S,R})
    copy!(d.f, s.f)
    copy!(d.z, s.z)
    d
end

"""
    vsize(A::ReMat)

Return the size of vector-valued random effects (i.e. 1 for a [`ScalarReMat`](@ref)).
"""
vsize(A::ReMat) = isa(A,ScalarReMat) ? 1 : size(A.z, 1)

"""
    nlevs(A::ReMat)

Return the number of levels in the grouping factor of `A`.
"""
nlevs(A::ReMat) = length(A.f.pool)

Base.size(A::ReMat) = (length(A.f), vsize(A) * nlevs(A))

Base.size(A::ReMat, i::Integer) =
    i < 1 ? throw(BoundsError()) : i == 1 ? length(A.f) :  i == 2 ? vsize(A)*nlevs(A) : 1

==(A::ReMat,B::ReMat) = (A.f == B.f) && (A.z == B.z)

function Base.A_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n,q = size(A)
    k = size(B, 2)
    if size(R, 1) ≠ n || size(B, 1) ≠ q || size(R, 2) ≠ k
        throw(DimensionMismatch())
    end
    if β ≠ 1
        β == 0 ? fill!(R, 0) : scale!(β, R)
    end
    rr, zz = A.f.refs, A.z
    if isa(A, ScalarReMat)
        for j in 1 : k, i in 1 : n
            R[i, j] += α * zz[i] * B[rr[i],j]
        end
    else
        l = size(zz,1)
        Bt = reshape(B, (l, div(q,l), k))
        for j in 1:k, i in 1:n
            R[i, j] += α * dot(Compat.view(Bt, :, Int(rr[i]), j), Compat.view(zz, :, i))
        end
    end
    R
end

Base.A_mul_B!{T}(A::ReMat, B::StridedVecOrMat{T}, R::StridedVecOrMat{T}) = A_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B!{T}(α::Real, A::ReMat, B::StridedVecOrMat{T}, β::Real, R::StridedVecOrMat{T})
    n, q = size(A)
    k = size(B, 2)
    if size(R, 1) ≠ q || size(B, 1) ≠ n || size(R, 2) ≠ k
        throw(DimensionMismatch())
    end
    if β ≠ 1
        β == 0 ? fill!(R, 0) : scale!(β, R)
    end
    rr, zz = A.f.refs, A.z
    if isa(A, ScalarReMat)
        for j in 1 : k, i in 1 : n
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

Base.Ac_mul_B!{T}(R::StridedVecOrMat{T}, A::ReMat, B::StridedVecOrMat{T}) = Ac_mul_B!(one(T), A, B, zero(T), R)

function Base.Ac_mul_B(A::ReMat, B::DenseVecOrMat)
    k = size(A, 2)
    Ac_mul_B!(zeros(eltype(B), isa(B, Vector) ? (k,) : (k, size(B, 2))), A, B)
end

function Base.Ac_mul_B(A::ScalarReMat, B::ScalarReMat)
    Az = A.z
    Ar = A.f.refs
    if is(A, B)
        v = zeros(eltype(Az), nlevs(A))
        for i in eachindex(Ar)
            v[Ar[i]] += abs2(Az[i])
        end
        return Diagonal(v)
    end
    densify(sparse(convert(Vector{Int32}, Ar), convert(Vector{Int32}, B.f.refs), Az .* B.z))
end

function Base.Ac_mul_B!{T}(C::Diagonal{T}, A::ScalarReMat{T}, B::ScalarReMat{T})
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

function Base.Ac_mul_B!{Tv, Ti}(C::SparseMatrixCSC{Tv, Ti}, A::ScalarReMat{Tv}, B::ScalarReMat{Tv})
    m, n = size(A)
    p, q = size(B)
    if m ≠ p || size(C, 1) ≠ n || size(C, 2) ≠ q
        throw(DimensionMismatch("size(A) = $(size(A)), size(B) = $(size(B)), size(C) = $(size(C))"))
    end
    SparseArrays.sparse!(convert(Vector{Ti}, A.f.refs), convert(Vector{Ti}, B.f.refs), A.z .* B.z,
        n, q, +, Array(Ti, q), Array(Ti, n + 1), Array(Ti, m), Array(Tv, m), C.colptr, C.rowval, C.nzval)
    C
end

function Base.Ac_mul_B!{T}(C::HBlkDiag{T}, A::VectorReMat{T}, B::VectorReMat{T})
    c, a, r = C.arr, A.z, A.f.refs
    _, m, n = size(c)
    fill!(c, 0)
    if !is(A, B)
        throw(ArgumentError("Currently defined only for A === B"))
    end
    for k in eachindex(r)
        ri = Int(r[k])
        for j in 1 : m
            aj = a[j, k]
            c[j, j, ri] += abs2(aj)
            for i in 1 : j - 1
                aij = a[i, k] * aj
                c[i, j, ri] += aij
                c[j, i, ri] += aij
            end
        end
    end
    C
end

function Base.Ac_mul_B!{T}(C::Matrix{T}, A::ScalarReMat{T}, B::ScalarReMat{T})
    m, n = size(C)
    ma, na = size(A)
    mb, nb = size(B)
    if m ≠ na || n ≠ nb || ma ≠ mb
        throw(DimensionMismatch())
    end
    a, b, ra, rb = A.z, B.z, A.f.refs, B.f.refs
    fill!(C, 0)
    for i in eachindex(a)
        C[ra[i], rb[i]] += a[i] * b[i]
    end
    C
end

function Base.Ac_mul_B{T}(A::VectorReMat{T}, B::VectorReMat{T})
    if is(A, B)
        l = size(A.z, 1)
        return Ac_mul_B!(HBlkDiag(Array(T, (l, l, length(A.f.pool)))), A, B)
    end
    Az = A.z
    Bz = B.z
    if (m = size(Az, 2)) ≠ size(Bz, 2)
        throw(DimensionMismatch("$m = size(Az,2) ≠ size(Bz,2) = $(size(Bz, 2))"))
    end
    a, b = size(Az,1), size(Bz, 1)
    ab = a * b
    nz = ab * m
    I, J, V = sizehint!(Int[], nz), sizehint!(Int[], nz), sizehint!(T[], nz)
    Ar, Br = A.f.refs, B.f.refs
    Ipat = Compat.repeat(1 : a, outer = b)
    Jpat = Compat.repeat(1 : b, inner = a)
    for i in 1 : m
        append!(I, Ipat + (Ar[i] - 1) * a)
        append!(J, Jpat + (Br[i] - 1) * b)
        append!(V, vec(Compat.view(Az, :, i) * Compat.view(Bz, :, i)'))
    end
    sparse(I, J, V)
end

function Base.Ac_mul_B!{T}(R::DenseVecOrMat{T}, A::DenseVecOrMat{T}, B::ReMat)
    m = size(A, 1)
    n = size(A, 2)
    p, q = size(B)
    if m ≠ p || size(R, 1) ≠ n || size(R, 2) ≠ q
        throw(DimensionMismatch(""))
    end
    fill!(R, 0)
    r, z = B.f.refs, B.z
    if isa(B, ScalarReMat)
        for j in 1 : n, i in 1 : m
            R[j, r[i]] += A[i, j] * z[i]
        end
    else
        l = size(zz, 1)
        for j in 1 : n, i in 1 : m
            roffset = (r[i] - 1) * l
            aij = A[i, j]
            for k in 1:l
                R[j, roffset + k] += aij * z[k, i]
            end
        end
    end
    R
end

Base.Ac_mul_B(A::DenseVecOrMat, B::ReMat) = Ac_mul_B!(Array(eltype(A), (size(A, 2), size(B, 2))), A, B)

(*){T}(D::Diagonal{T}, A::ScalarReMat{T}) = ScalarReMat(A.f, D * A.z, A.fnm, A.cnms)

function Base.A_mul_B!{T}(C::ScalarReMat{T}, A::Diagonal{T}, B::ScalarReMat{T})
    map!(*, C.z, A.diag, B.z)
    C
end

function Base.A_mul_B!{T}(A::Diagonal{T}, B::ScalarReMat{T})
    a, b = A.diag, B.z
    if length(a) ≠ length(b)
        throw(DimensionMismatch("A_mul_B!, A: diagonal $(size(A, 1)), B: ScalarReMat $(size(B))"))
    end
    for i in eachindex(a)
        b[i] *= a[i]
    end
    B
end

function Base.A_mul_B!{T}(A::Diagonal{T}, B::VectorReMat{T})
    a, b = A.diag, B.z
    if length(a) ≠ size(b, 2)
        throw(DimensionMismatch("A_mul_B!, A: diagonal $(size(A, 1)), B: ScalarReMat $(size(B))"))
    end
    k = size(b, 1)
    for j in eachindex(a)
        aj = a[j]
        for i in 1 : k
            b[i, j] *= aj
        end
    end
    B
end

(*){T}(D::Diagonal{T}, A::VectorReMat{T}) = VectorReMat(A.f, A.z * D, A.fnm, A.cnms)

function Base.A_mul_B!{T}(C::VectorReMat{T}, A::Diagonal{T}, B::VectorReMat{T})
    A_mul_B!(C.z, B.z, A)
    C
end
