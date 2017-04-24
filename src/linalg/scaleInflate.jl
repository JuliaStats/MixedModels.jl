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
