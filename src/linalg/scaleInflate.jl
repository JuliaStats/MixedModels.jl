"""
    scaleInflate!(L::AbstractMatrix, A::AbstractMatrix, Λ::AbstractTerm)

Overwrite a diagonal block of `L` with the corresponding block of `Λ'AΛ + I` except when Λ
is a [`MatrixTerm`]{@ref}, in which case this becomes `copyto!(L, A)`.
"""
function scaleInflate! end

function scaleInflate!(Ljj::Matrix{T}, Ajj::Matrix{T}, Λj::MatrixTerm{T}) where {T}
    @argcheck(size(Ljj) == size(Ajj), DimensionMismatch)
    copyto!(Ljj, Ajj)
end

function scaleInflate!(Ljj::Diagonal{T}, Ajj::Diagonal{T},
        Λj::ScalarFactorReTerm{T}) where {T}
    broadcast!((x,k) -> k * x + one(T), Ljj.diag, Ajj.diag, abs2(Λj.Λ))
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::Diagonal{T},
        Λj::ScalarFactorReTerm{T}) where {T}
    n = LinearAlgebra.checksquare(Ljj)
    Ad = Ajj.diag
    @argcheck(length(Ad) == n, DimensionMismatch)
    lambsq = abs2(Λj.Λ)
    fill!(Ljj, zero(T))
    for (j, dj) in enumerate(Ad)
        Ljj[j,j] = lambsq * dj + one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::UniformBlockDiagonal{T}, Ajj::UniformBlockDiagonal{T},
        Λj::VectorFactorReTerm{T}) where {T}
    Ljjdd = Ljj.data
    Ajjdd = Ajj.data
    @argcheck(size(Ljjdd) == size(Ajjdd), DimensionMismatch)
    copyto!(Ljjdd, Ajjdd)
    λ = Λj.Λ
    for Lf in Ljj.facevec
        lmul!(adjoint(λ), rmul!(Lf, λ))
        for d in diagind(Lf)
            Lf[d] += one(T)
        end
    end
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::UniformBlockDiagonal{T},
        Λj::VectorFactorReTerm{T}) where {T}
    @argcheck(size(Ljj) == size(Ajj), DimensionMismatch)
    λ = Λj.Λ
    Afv = Ajj.facevec
    m, n, l = size(Ajj.data)
    m == n || throw(ArgumentError("Diagonal blocks of Ajj must be square"))
    fill!(Ljj, zero(T))
    tmp = Array{T}(undef, m, m)
    for (k, Af) in enumerate(Afv)
        lmul!(adjoint(λ), rmul!(copyto!(tmp, Af), λ))
        offset = (k - 1)*m
        for j in 1:m, i in 1:m
            Ljj[offset + i, offset + j] = tmp[i, j] + (i == j)
        end
    end
    Ljj
end
