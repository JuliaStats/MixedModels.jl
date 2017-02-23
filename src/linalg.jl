function cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:L}})
    map!(sqrt, D.diag)
    D
end

cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:U}}) = cholUnblocked!(D, Val{:L})

function cholUnblocked!{T<:BlasFloat}(A::Matrix{T}, ::Type{Val{:L}})
    _, info = LAPACK.potrf!('L', A)
    if info ≠ 0
        throw(PosDefException(info))
    end
    A
end

function cholUnblocked!{T<:AbstractMatrix}(D::Diagonal{T}, ::Type{Val{:L}})
    map!(b -> cholUnblocked!(b, Val{:L}), D.diag)
    D
end

function rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, β::T, C::Hermitian{T,S})
    m, n = size(A)
    if m ≠ LinAlg.checksquare(C) || C.uplo != 'L'
        throw(DimensionMismatch("C is not Hermitian lower or size(C, 2) ≠ size(A, 1)"))
    end
    Cd = C.data
    if β ≠ one(T)
        scale!(LowerTriangular(Cd), β)
    end
    rv, nz = rowvals(A), nonzeros(A)
    @inbounds for jj in 1:n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            anzj, rvj = α * nz[j], rv[j]
            for i in k:lenrngjj
                kk = rangejj[i]
                Cd[rv[kk], rvj] += nz[kk] * anzj
            end
        end
    end
    C
end

rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, C::Hermitian{T,S}) = rankUpdate!(α, A, one(T), C)

function rankUpdate!{T <: Number}(α::T, A::SparseMatrixCSC{T}, C::Diagonal{T})
    m, n = size(A)
    dd = C.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(C,2) = $(length(dd)) ≠ $m = size(A,1)"))
    end
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1 : n
        nzr = nzrange(A, j)
        length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
        k = nzr[1]
        @inbounds dd[rv[k]] += α * abs2(nz[k])
    end
    return C
end

function cond{T}(J::UniformScaling{T})
    onereal = inv(one(real(J.λ)))
    return J.λ ≠ zero(T) ? onereal : oftype(onereal, Inf)
end

A_mul_B!{T}(A::UniformScaling{T}, B::AbstractArray{T}) = scale!(B, A.λ)

A_mul_B!{T}(A::AbstractArray{T}, B::UniformScaling{T}) = scale!(A, B.λ)

function A_mul_B!{T}(A::Diagonal{T}, B::UniformScaling{T})
    scale!(A.diag, B.λ)
    A
end

function A_mul_B!(A::Diagonal, B::UniformSc)
    for a in A.diag
        A_mul_B!(a, B.λ)
    end
    A
end

function A_mul_B!{T<:AbstractMatrix}(A::Matrix, B::UniformSc{T})
    λ = B.λ
    k = size(λ, 1)
    m, n = size(A)
    q, r = divrem(n, k)
    if r ≠ 0
        throw(DimensionMismatch("size(A, 2) = $n is not a multiple of size(B.λ, 1) = $k"))
    end
    offset = 0
    onetok = 1:k
    for blk in 1:q
        A_mul_B!(view(A, :, onetok + offset), λ)
        offset += k
    end
    A
end

function A_mul_B!{T}(C::StridedVecOrMat{T}, A::UniformScaling{T}, B::StridedVecOrMat{T})
    if size(C) ≠ size(B)
        throw(DimensionMismatch("size(C) = $(size(C)) ≠ $(size(B)) = size(B)"))
    end
    broadcast!(*, C, A.λ, B)
end

function A_mul_Bc!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T},
    β::T, C::StridedMatrix{T})
    BLAS.gemm!('N', 'C', α, A, B, β, C)
end

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T},
    β::T, C::StridedVecOrMat{T})
    if size(B, 1) ≠ (n = size(C, 2))
        throw(DimensionMismatch("size(B, 1) = $(size(B, 1)) ≠ $n = size(C, 2)"))
    end
    nzv = A.nzval
    rv = A.rowval
    if β ≠ one(T)
        β ≠ zero(0) ? scale!(C, β) : fill!(C, β)
    end
    for col = 1:A.n
        for k = 1:n
            αxj = α*B[k, col]
            @inbounds for j = nzrange(A, col)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end

function A_mul_Bc!{T<:Number}(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T},
    β::T, C::StridedVecOrMat{T})
    m, n = size(A)
    p, q = size(B)
    r, s = size(C)
    if r ≠ m || s ≠ p || n ≠ q
        throw(DimensionMismatch("size(C,1) ≠ size(A,1) or size(C,2) ≠ size(B,1) or size(A,2) ≠ size(B,2)"))
    end
    if β ≠ one(T)
        β ≠ zero(T) ? scale!(C, β) : fill!(C, β)
    end
    nz, rv = nonzeros(B), rowvals(B)
    @inbounds for j in 1:q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1:r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    C
end

function Ac_mul_B!{T}(A::UniformScaling{T}, B::Diagonal{T})
    scale!(B.diag, A.λ)
    B
end

Ac_mul_B!{T}(A::UniformScaling{T}, B::AbstractArray{T}) = scale!(B, A.λ)

function Ac_mul_B!(A::UniformSc, B::Diagonal)
    for b in B.diag
        Ac_mul_B!(A.λ, b)
    end
    B
end

function Ac_mul_B!{T<:AbstractMatrix}(A::UniformSc{T}, B::Matrix)
    λ = A.λ
    k = size(λ, 1)
    m, n = size(B)
    q, r = divrem(m, k)
    if r ≠ 0
        throw(DimensionMismatch("size(B, 1) = $m is not a multiple of size(A.λ, 1) = $k"))
    end
    offset = 0
    onetok = 1:k
    for blk in 1:q
        blkrows = onetok + offset
        for j in 1:n
            Ac_mul_B!(λ, view(A, blkrows, j))
        end
        offset += k
    end
    B
end

Ac_ldiv_B!{T}(D::Diagonal{T}, B) = A_ldiv_B!(D, B)

function A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    if size(D) ≠ size(B)
        throw(DimensionMismatch("size(D) ≠ size(B)"))
    end
    map!(/, B.diag, B.diag, D.diag)
    B
end

function A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    dd = D.diag
    if length(dd) ≠ size(B, 1)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    vals = nonzeros(B)
    rows = rowvals(B)
    @inbounds for k in eachindex(vals)
        vals[k] /= dd[rows[k]]
    end
    B
end

function A_rdiv_B!{T}(A::StridedMatrix{T}, D::Diagonal{T})
    scale!(A, inv.(D.diag))
    A
end

A_rdiv_Bc!{T}(A::StridedMatrix{T}, D::Diagonal{T}) = LinAlg.A_rdiv_B!(A, D)

function A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, D::Diagonal{T})
    m,n = size(A)
    dd = D.diag
    if length(dd) ≠ n
        throw(DimensionMismatch("size(A, 2) = $n ≠ size(D, 2) = $(length(dd))"))
    end
    nonz = nonzeros(A)
    for j in 1 : n
        ddj = dd[j]
        for k in nzrange(A, j)
            nonz[k] /= ddj
        end
    end
    A
end

# FIXME: This function should not call LowerTriangular
function A_rdiv_Bc!{T<:AbstractMatrix}(A::Matrix, B::Diagonal{T})
    offset = 0
    for d in B.diag
        k = size(d, 1)
        A_rdiv_B!(view(A, :, (1:k) + offset), LowerTriangular(d))
        offset += k
    end
    A
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [norm(view(ld, i, 1:i)) for i in 1 : size(ld, 1)]
end

rowlengths(L::UniformScaling) = [abs(L.λ)]
