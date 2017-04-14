"""
    cholUnblocked!(A, Val{:L})

Overwrite the lower triangle of `A` with its lower Cholesky factor.

The name is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
because these are part of the inner calculations in a blocked Cholesky factorization.
"""
function cholUnblocked! end

function cholUnblocked!{T<:AbstractFloat}(D::Diagonal{T}, ::Type{Val{:L}})
    map!(sqrt, D.diag, D.diag)
    D
end

cholUnblocked!{T<:AbstractFloat}(D::Diagonal{T}, ::Type{Val{:U}}) = cholUnblocked!(D, Val{:L})

function cholUnblocked!{T<:AbstractFloat}(A::Diagonal{Matrix{T}}, ::Type{Val{:L}})
    map!(m -> cholUnblocked!(m, Val{:L}), A.diag)
    A
end
function cholUnblocked!{T<:BlasFloat}(A::Matrix{T}, ::Type{Val{:L}})
    n = checksquare(A)
    if n == 1
        A[1] < zero(T) && throw(PosDefException(1))
        A[1] = sqrt(A[1])
    elseif n == 2
        A[1] = sqrt(A[1])
        A[2] /= A[1]
        A[4] = sqrt(A[4] - abs2(A[2]))
    else
        _, info = LAPACK.potrf!('L', A)
        info ≠ 0 && throw(PosDefException(info))
    end
    A
end

function cholUnblocked!{T<:AbstractFloat}(D::Diagonal{LowerTriangular{T, Matrix{T}}},
    ::Type{Val{:L}})
    for b in D.diag
        cholUnblocked!(b.data, Val{:L})
    end
    D
end

"""
    rankUpdate!(A, C)
    rankUpdate!(α, A, C)
    rankUpdate!(α, A, β, C)

A rank-k update of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
"""
function rankUpdate! end

rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, a::StridedVector{T}, A::HermOrSym{T,S}) = BLAS.syr!(A.uplo, α, a, A.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(a::StridedVector{T}, A::HermOrSym{T,S}) = rankUpdate!(one(T), a, A)

rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, β::T, C::HermOrSym{T,S}) = BLAS.syrk!(C.uplo, 'N', α, A, β, C.data)
rankUpdate!{T<:Real,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, C::HermOrSym{T,S}) = rankUpdate!(α, A, one(T), C)
rankUpdate!{T<:Real,S<:StridedMatrix}(A::StridedMatrix{T}, C::HermOrSym{T,S}) = rankUpdate!(one(T), A, one(T), C)

function rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, β::T, C::HermOrSym{T,S})
    m, n = size(A)
    @argcheck m == size(C, 2) && C.uplo == 'L' DimensionMismatch
    Cd = C.data
    if β ≠ one(T)
        scale!(LowerTriangular(Cd), β)
    end
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for jj in 1:n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            anzj = α * nz[j]
            rvj = rv[j]
            for i in k:lenrngjj
                kk = rangejj[i]
                Cd[rv[kk], rvj] += nz[kk] * anzj
            end
        end
    end
    C
end

rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, C::HermOrSym{T,S}) = rankUpdate!(α, A, one(T), C)

function rankUpdate!{T <: Number}(α::T, A::SparseMatrixCSC{T}, C::Diagonal{T})
    m, n = size(A)
    dd = C.diag
    @argcheck length(dd) == m DimensionMismatch
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1:n
        nzr = nzrange(A, j)
        length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
        k = nzr[1]
        @inbounds dd[rv[k]] += α * abs2(nz[k])
    end
    C
end

function rankUpdate!{T<:Number}(α::T, A::SparseMatrixCSC{T}, C::Diagonal{LowerTriangular{T,Matrix{T}}})
    m, n = size(A)
    cdiag = C.diag
    dsize = size.(cdiag, 2)
    @argcheck sum(dsize) == m DimensionMismatch
    if all(dsize .== 1)
        nz = nonzeros(A)
        rv = rowvals(A)
        for j in 1:n
            nzr = nzrange(A, j)
            length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
            k = nzr[1]
            @inbounds cdiag[rv[k]].data[1] += α * abs2(nz[k])
        end
    else  # not efficient but only used for nested vector-valued r.e.'s, which are rare
        aat = α * (A * A')
        nz = nonzeros(aat)
        rv = rowvals(aat)
        offset = 0
        for d in cdiag
            k = size(d, 2)
            for j in 1:k
                for i in nzrange(aat, offset + j)
                    ii = rv[i] - offset
                    0 < ii ≤ k || throw(ArgumentError("A*A' does not conform to B"))
                    if ii ≥ j  # update lower triangle only
                        d.data[ii, j] += nz[i]
                    end
                end
            end
            offset += k
        end
    end
    C
end

"""
    scaleInflate!(L, A, Λ)

Overwrite a diagonal block of `L` with the corresponding block of `Λ'AΛ + I`
"""
function scaleInflate! end

function scaleInflate!{T<:AbstractFloat}(Ljj::Diagonal{T}, Ajj::Diagonal{T}, Λj::UniformScaling{T})
    broadcast!((x,k) -> k * x + one(T), Ljj.diag, Ajj.diag, abs2(Λj.λ))
    Ljj
end

function scaleInflate!{T<:AbstractFloat}(Ljj::Matrix{T}, Ajj::Diagonal{T}, Λj::UniformScaling{T})
    Ad = Ajj.diag
    @argcheck length(Ad) == checksquare(Ljj) DimensionMismatch
    lambsq = abs2(Λj.λ)
    fill!(Ljj, zero(T))
    for (j, jj) in zip(eachindex(Ad), diagind(Ljj))
        Ljj[jj] = lambsq * Ad[j] + one(T)
    end
    Ljj
end

function scaleInflate!{T}(Ljj::Matrix{T}, Ajj::Matrix{T}, Λj::Identity{T})
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    copy!(Ljj, Ajj)
end

function scaleInflate!{T<:AbstractFloat}(Ljj::Diagonal{LowerTriangular{T,Matrix{T}}},
    Ajj::Diagonal{Matrix{T}}, Λj::MaskedLowerTri{T})
    λ = Λj.m
    Ldiag = Ljj.diag
    Adiag = Ajj.diag
    nblk = length(Ldiag)
    @argcheck length(Adiag) == length(Ldiag)
    for i in eachindex(Ldiag)
        Ldi = Ac_mul_B!(λ, A_mul_B!(copy!(Ldiag[i].data, Adiag[i]), λ))
        for k in diagind(Ldi)
            Ldi[k] += one(T)
        end
    end
    Ljj
end

function scaleInflate!{T<:AbstractFloat}(Ljj::Matrix{T}, Ajj::Diagonal{Matrix{T}},
    Λj::MaskedLowerTri{T})
    Adiag = Ajj.diag
    λ = Λj.m
    n = size(λ, 2)
    @argcheck all(a -> size(a) == (n, n), Adiag) && size(Ljj, 2) == sum(size.(Adiag, 2))
    fill!(Ljj, zero(T))
    scrm = Matrix{T}(n, n)
    offset = 0
    for a in Adiag
        Ac_mul_B!(λ, A_mul_B!(copy!(scrm, a), λ))
        for j in 1:n, i in 1:n
            Ljj[offset + i, offset + j] = scrm[i, j] + T(i == j)
        end
        offset += n
    end
    Ljj
end

function A_mul_Bc!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T},
    β::T, C::StridedMatrix{T})
    BLAS.gemm!('N', 'C', α, A, B, β, C)
end

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T},
    β::T, C::StridedVecOrMat{T})
    n = size(B, 1)
    @argcheck size(C, 2) == n DimensionMismatch
    nzv = A.nzval
    rv = A.rowval
    if β ≠ one(T)
        β ≠ zero(T) ? scale!(C, β) : fill!(C, β)
    end
    for col = 1:A.n
        for k = 1:n
            αxk = α * B[k, col]
            @inbounds for j = nzrange(A, col)
                C[rv[j], k] += nzv[j] * αxk
            end
        end
    end
    C
end

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
    β::T, C::Matrix{T})
    @argcheck B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n  DimensionMismatch
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    if β ≠ one(T)
        β ≠ zero(T) ? scale!(C, β) : fill!(C, β)
    end
    for j = 1:A.n
        for ib in nzrange(B, j)
            αbnz = α * bnz[ib]
            jj = brv[ib]
            for ia in nzrange(A, j)
                C[arv[ia], jj] += anz[ia] * αbnz
            end
        end
    end
    C
end

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
    β::T, C::SparseMatrixCSC{T})
    @argcheck B.m == C.n && A.m == C.m && A.n == B.n  DimensionMismatch
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    cnz = nonzeros(C)
    crv = rowvals(C)
    if β ≠ one(T)
        iszero(β) ? fill!(cnz, β) : scale!(cnz, β)
    end
    for j = 1:A.n
        for ib in nzrange(B, j)
            αbnz = α * bnz[ib]
            jj = brv[ib]
            for ia in nzrange(A, j)
                crng = nzrange(C, jj)
                ind = findfirst(crv[crng], arv[ia])
                if iszero(ind)
                    throw(ArgumentError("A*B' has nonzero positions not in C"))
                end
                cnz[crng[ind]] += anz[ia] * αbnz
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
        iszero(β) ? fill!(C, β) : scale!(C, β)
    end
    nz = nonzeros(B)
    rv = rowvals(B)
    @inbounds for j in 1:q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1:r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    C
end

Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = BLAS.gemm!('C', 'N', α, A, B, β, C)

Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedVector{T}, β::T, C::StridedVector{T}) = BLAS.gemv!('C', α, A, B, β, C)

function Ac_ldiv_B!{T<:AbstractFloat}(A::Diagonal{LowerTriangular{T,Matrix{T}}}, B::StridedVector{T})
    offset = 0
    for a in A.diag
        k = size(a, 1)
        Ac_ldiv_B!(a, view(B, (1:k) + offset))
        offset += k
    end
    B
end

Ac_ldiv_B!{T}(D::Diagonal{T}, B::StridedVecOrMat{T}) = A_ldiv_B!(D, B)

function A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    @argcheck size(D) == size(B) DimensionMismatch
    map!(/, B.diag, B.diag, D.diag)
    B
end

function A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    @argcheck size(D, 2) == size(B, 1) DimensionMismatch
    dd = D.diag
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

function A_rdiv_B!{T}(A::StridedMatrix{T}, D::Diagonal{LowerTriangular{T, Matrix{T}}})
    offset = 0
    for L in D.diag
        k = size(L, 1)
        A_rdiv_B!(view(A, :, (1:k) + offset), L)
        offset += k
    end
    A
end

A_rdiv_Bc!{T}(A::StridedMatrix{T}, D::Diagonal{T}) = LinAlg.A_rdiv_B!(A, D)

function A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, D::Diagonal{T})
    @argcheck size(D, 2) == size(A, 2) DimensionMismatch
    dd = D.diag
    nonz = nonzeros(A)
    for j in 1:A.n
        ddj = dd[j]
        for k in nzrange(A, j)
            nonz[k] /= ddj
        end
    end
    A
end

function A_rdiv_Bc!{T<:AbstractFloat}(A::Matrix, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    offset = 0
    for d in B.diag
        k = size(d, 1)
        A_rdiv_Bc!(view(A, :, (1:k) + offset), d)
        offset += k
    end
    A
end

function A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    nz = nonzeros(A)
    offset = 0
    for d in B.diag
        if (k = size(d, 1)) == 1
            d1 = d[1]
            offset += 1
            for k in nzrange(A, offset)
                nz[k] /= d1
            end
        else
            nzr = nzrange(A, offset + 1).start : nzrange(A, offset + k).stop
            q = div(length(nzr), k)
            A_rdiv_Bc!(reshape(view(nz, nzr), (q, k)), d)
            offset += k
        end
    end
    A
end

function full{T}(A::Diagonal{LowerTriangular{T,Matrix{T}}})
    D = diag(A)
    sz = size.(D, 2)
    n = sum(sz)
    B = Array{T}((n,n))
    offset = 0
    for (d,s) in zip(D, sz)
        for j in 1:s, i in j:s
            B[offset + i, offset + j] = d[i,j]
        end
        offset += s
    end
    B
end

function rowlengths{T}(Λ::MaskedLowerTri{T})
    ld = Λ.m.data
    [norm(view(ld, i, 1:i)) for i in 1:size(ld, 1)]
end

rowlengths(L::UniformScaling) = [abs(L.λ)]
