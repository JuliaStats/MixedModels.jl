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

function scaleInflate!(Ljj::LowerTriangular{T,UniformBlockDiagonal{T}},
                       Ajj::UniformBlockDiagonal{T},
                       Λj::VectorFactorReTerm{T}) where {T}
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    λ = LowerTriangular(Λj.Λ)
    k = vsize(Λj)
    for (Lf, Af) in zip(Ljj.data.facevec, Ajj.facevec)
        Ac_mul_B!(λ, A_mul_B!(copy!(Lf, Af), λ))
        for j in 1:k
            Lf[j, j] += one(T)
        end
    end
    Ljj
end

function scaleInflate!(Ljj::LowerTriangular{T,Matrix{T}}, Ajj::UniformBlockDiagonal{T},
                       Λj::VectorFactorReTerm{T}) where {T}
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    λ = LowerTriangular(Λj.Λ)
    Afv = Ajj.facevec
    m, n, l = size(Ajj.data)
    m == n || throw(ArgumentError("Diagonal blocks of Ajj must be square"))
    Ld = Ljj.data
    fill!(Ld, zero(T))
    tmp = Array{T}(m, m)
    offset = 0
    for k in eachindex(Afv)
        Ac_mul_B!(λ, A_mul_B!(copy!(tmp, Afv[k]), λ))
        for j in 1:n, i in 1:n
            Ld[offset + i, offset + j] = tmp[i, j] + T(i == j)
        end
        offset += m
    end
    Ljj
end
