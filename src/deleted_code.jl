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

function Base.A_mul_B!{T}(C::ScalarReMat{T}, A::Diagonal{T}, B::ScalarReMat{T})
    map!(*, C.z, A.diag, B.z)
    C
end

function unscaledre!{T}(y::AbstractVector{T}, M::ScalarReMat{T}, L::UniformScaling{T})
    re = randn(1, length(M.f.pool))
    unscaledre!(y, M, (re *= L))
end

"""
    unscaledre!{T}(y::Vector{T}, M::ReMat{T}, b::Matrix{T})

Add unscaled random effects defined by `M` and `b` to `y`.
"""
function unscaledre!{T<:AbstractFloat}(y::Vector{T}, M::ScalarReMat{T}, b::Matrix{T})
    z = M.z
    @argcheck length(y) == length(z) && size(b, 1) == 1 DimensionMismatch
    inds = M.f.refs
    @inbounds for i in eachindex(y)
        y[i] += b[inds[i]] * z[i]
    end
    y
end
