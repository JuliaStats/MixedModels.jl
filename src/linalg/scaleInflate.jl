"""
    scaleInflate!(L::AbstractMatrix, A::AbstractMatrix, Λ::AbstractTerm)

Overwrite a diagonal block of `L` with the corresponding block of `Λ'AΛ + I`
"""
function scaleInflate! end

function scaleInflate!(Ljj::Diagonal{T}, Ajj::Diagonal{T}, Λj::ReMat{T,R,1}) where {T,R}
    broadcast!((x,k) -> k * x + one(T), Ljj.diag, Ajj.diag, abs2(Λj.λ[1]))
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::Diagonal{T}, Λj::ReMat{T,R,1}) where {T,R}
    n = LinearAlgebra.checksquare(Ljj)
    Ad = Ajj.diag
    length(Ad) == n || throw(DimensionMismatch(""))
    lambsq = abs2(Λj.λ[1])
    fill!(Ljj, zero(T))
    for (j, dj) in enumerate(Ad)
        Ljj[j,j] = lambsq * dj + one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::UniformBlockDiagonal{T}, Ajj::UniformBlockDiagonal{T},
        Λj::ReMat{T}) where {T}
    Ljjdd = Ljj.data
    Ajjdd = Ajj.data
    if ((m, n, l) = size(Ljjdd)) != size(Ajjdd) 
        throw(DimensionMismatch("size(Ljj.data) = $(size(Ljjdd)) != $(size(Ajjdd)) = size(Ajj.data)"))
    end
    copyto!(Ljjdd, Ajjdd)
    m, n, l = size(Ljjdd)
    λ = Λj.λ
    Lfv = Ljj.facevec
    dind = diagind(Lfv[1])
    @inbounds for Lf in Lfv
        lmul!(adjoint(λ), rmul!(Lf, λ))
    end
    @inbounds for k in 1:l, i in 1:m
        Ljjdd[i, i, k] += one(T)
    end
    Ljj
end

function scaleInflate!(Ljj::Matrix{T}, Ajj::UniformBlockDiagonal{T},
        Λj::ReMat{T}) where {T}
    if size(Ljj) != size(Ajj)
        throw(DimensionMismatch("size(Ljj) = $(size(Ljj)) != $(size(Ajj)) = size(Ajj)"))
    end
    λ = Λj.λ
    Afv = Ajj.facevec
    m, n, l = size(Ajj.data)
    m == n || throw(ArgumentError("Diagonal blocks of Ajj must be square"))
    fill!(Ljj, zero(T))
    tmp = Array{T}(undef, m, m)
    @inbounds for (k, Af) in enumerate(Afv)
        lmul!(adjoint(λ), rmul!(copyto!(tmp, Af), λ))
        offset = (k - 1)*m
        for j in 1:m, i in 1:m
            Ljj[offset + i, offset + j] = tmp[i, j] + (i == j)
        end
    end
    Ljj
end
