"""
    scaleInflate!(L::AbstractMatrix, A::AbstractMatrix, Λ::AbstractTerm)

Overwrite a diagonal block of `L` with the corresponding block of `Λ'AΛ + I` except when Λ
is a [`MatrixTerm`]{@ref}, in which case this becomes `copyto!(L, A)`.
"""
function scaleInflate! end

function scaleInflate!(Ljj::Matrix{T}, Ajj::Matrix{T}, Λj::MatrixTerm{T}) where T
    @argcheck(size(Ljj) == size(Ajj), DimensionMismatch)
    copyto!(Ljj, Ajj)
end

function scaleInflate!(Ljj::Diagonal{T}, Ajj::Diagonal{T},
                       Λj::ScalarFactorReTerm{T}) where T<:AbstractFloat
    broadcast!((x,k) -> k * x + one(T), Ljj.diag, Ajj.diag, abs2(Λj.Λ))
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::Diagonal{T}, Λj::ScalarFactorReTerm{T}) where T
    Ad = Ajj.diag
    @argcheck(length(Ad) == size(Ljj, 1), DimensionMismatch)
    lambsq = abs2(Λj.Λ)
    fill!(Ljj, zero(T))
    for j in eachindex(Ad)
        Ljj[j,j] = lambsq * Ad[j] + one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::UniformBlockDiagonal{T}, Ajj::UniformBlockDiagonal{T},
                       Λj::VectorFactorReTerm{T}) where T
    @argcheck(size(Ljj) == size(Ajj), DimensionMismatch)
    Ljjdd = Ljj.data
    copyto!(Ljjdd, Ajj.data)
    k, m, n = size(Ljjdd)
    λ = Λj.Λ
    for Lf in Ljj.facevec
        lmul!(adjoint(λ), rmul!(Lf, λ))
    end
    for j in 1:n, i in 1:k
        Ljjdd[i, i, j] += one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::UniformBlockDiagonal{T},
                       Λj::VectorFactorReTerm{T}) where T
    @argcheck size(Ljj) == size(Ajj) DimensionMismatch
    λ = Λj.Λ
    Afv = Ajj.facevec
    m, n, l = size(Ajj.data)
    m == n || throw(ArgumentError("Diagonal blocks of Ajj must be square"))
    fill!(Ljj, zero(T))
    tmp = Array{T}(undef, m, m)
    offset = 0
    for k in eachindex(Afv)
        lmul!(adjoint(λ), rmul!(copyto!(tmp, Afv[k]), λ))
        for j in 1:n, i in 1:n
            Ljj[offset + i, offset + j] = tmp[i, j] + T(i == j)
        end
        offset += m
    end
    Ljj
end
