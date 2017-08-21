"""
    scaleInflate!(L::AbstractMatrix, A::AbstractMatrix, Λ::AbstractTerm)

Overwrite a diagonal block of `L` with the corresponding block of `Λ'AΛ + I` except when Λ
is a [`MatrixTerm`]{@ref}, in which case this becomes `copy!(L, A)`.
"""
function scaleInflate! end

function scaleInflate!(Ljj::LowerTriangular{T,Matrix{T}}, Ajj::Matrix{T},
                       Λj::MatrixTerm{T}) where T
    @argcheck(size(Ljj) == size(Ajj), DimensionMismatch)
    copy!(Ljj.data, Ajj)
end

function scaleInflate!(Ljj::Diagonal{T}, Ajj::Diagonal{T},
                     Λj::ScalarFactorReTerm{T}) where T<:AbstractFloat
    @argcheck(length(Λj.Λ) == 1, DimensionMismatch)
    broadcast!((x,k) -> k * x + one(T), Ljj.diag, Ajj.diag, abs2(Λj.Λ[1]))
    Ljj
end

function scaleInflate!(Ljj::LowerTriangular{T,Matrix{T}}, Ajj::Diagonal{T},
                       Λj::ScalarFactorReTerm{T}) where T
    Ldat = Ljj.data
    Ad = Ajj.diag
    @argcheck(length(Ad) == size(Ldat, 1), DimensionMismatch)
    lambsq = abs2(Λj.Λ)
    fill!(Ldat, zero(T))
    for (j, jj) in zip(eachindex(Ad), diagind(Ldat))
        Ldat[jj] = lambsq * Ad[j] + one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::LowerTriangular{T,UniformBlockDiagonal{T,K,L}},
                       Ajj::UniformBlockDiagonal{T,K,L},
                       Λj::VectorFactorReTerm{T}) where {T,K,L}
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    λ = LowerTriangular(Λj.Λ)
    Ldiag = Ljj.data.data
    Adiag = Ajj.data
    for i in eachindex(Ldiag)
        Ldi = Ac_mul_B!(λ, A_mul_B!(copy!(Ldiag[i], Adiag[i]), λ))
        for k in diagind(Ldi)
            Ldi[k] += one(T)
        end
    end
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::UniformBlockDiagonal{T},
                       Λj::VectorFactorReTerm{T}) where T<:AbstractFloat
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    Adiag = Ajj.data
    λ = LowerTriangular(Λj.Λ)
    n = size(λ, 2)
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
