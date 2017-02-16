function Base.A_mul_B!{T}(C::StridedVecOrMat{T}, A::UniformScaling{T}, B::StridedVecOrMat{T})
    if size(C) ≠ size(B)
        throw(DimensionMismatch("size(C) = $(size(C)) ≠ $(size(B)) = size(B)"))
    end
    C .= A.λ .* B
end

function cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:L}})
    map!(sqrt, D.diag)
    D
end

cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:U}}) = cholUnblocked!(D, Val{:L})

A_mul_Bc!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) =
    Base.BLAS.gemm!('N', 'C', α, A, B, β, C)

Base.A_mul_B!{T}(A::UniformScaling{T}, B::StridedVecOrMat{T}) = A_mul_B!(B, A, B)

LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T}, B) = A_ldiv_B!(D, B)

function LinAlg.A_rdiv_Bc!{T}(A::StridedMatrix{T}, D::Diagonal{T})
    m,n = size(A)
    dd = D.diag
    if length(dd) ≠ n
        throw(DimensionMismatch("size(A, 2) = $n ≠ size(D, 2) = $(length(dd))"))
    end
    @inbounds for j in 1 : n
        ddj = dd[j]
        for i in 1 : m
            A[i, j] /= ddj
        end
    end
    A
end

function LinAlg.A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, D::Diagonal{T})
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
    for jj in 1:n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            nzj, rvj = nz[j], rv[j]
            for i in k:lenrngjj
                kk = rangejj[i]
                Cd[rv[kk], rvj] += α * nz[kk] * nzj
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

LinAlg.A_mul_B!{T}(A::AbstractArray{T}, B::UniformScaling{T}) = scale!(A, B.λ)
LinAlg.Ac_mul_B!{T}(A::UniformScaling{T}, B::AbstractArray{T}) = scale!(B, A.λ)

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    if size(D) ≠ size(B)
        throw(DimensionMismatch("size(D) ≠ size(B)"))
    end
    map!(/, B.diag, B.diag, D.diag)
    B
end

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
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

function Base.LinAlg.A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T},
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

function Base.LinAlg.A_mul_Bc!{T<:Number}(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T},
    β::T, C::StridedVecOrMat{T})
    (m, n), (p, q), (r, s) = size(A), size(B), size(C)
    if r ≠ m || s ≠ p || n ≠ q
        throw(DimensionMismatch("size(C,1) ≠ size(A,1) or size(C,2) ≠ size(B,1) or size(A,2) ≠ size(B,2)"))
    end
    if β ≠ one(T)
        β ≠ zero(T) ? scale!(C, β) : fill!(C, β)
    end
    nz, rv = nonzeros(B), rowvals(B)
    for j in 1:q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1:r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    return C
end

function cholUnblocked!{T<:Union{Float32,Float64}}(A::Matrix{T}, ::Type{Val{:L}})
    _, info = LAPACK.potrf!('L', A)
    if info ≠ 0
        throw(LinAlg.PosDefException(info))
    end
    A
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [norm(view(ld, i, 1:i)) for i in 1 : size(ld, 1)]
end

rowlengths(L::UniformScaling) = [abs(L.λ)]
