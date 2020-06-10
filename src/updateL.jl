"""
    λdot(re::ReMat{T,S}, ind::Integer) where {T,S}

Return the lower-triangular indicator matrix for the `ind`th element of θ in λ
"""
function λdot end

function λdot(re::ReMat{T,1}, ind::Integer) where {T}
    @assert isone(ind)
    LowerTriangular(fill!(view(re.scratch, :, 1:1), one(T)))
end

function λdot(re::ReMat{T,S}, ind::Integer) where {T,S}
    value = fill!(view(re.scratch, :, axes(re.scratch, 1)), zero(T))
    value[re.inds[ind]] = one(T)
    LowerTriangular(value)
end

"""
    scaleinflate!(L::AbstractMatrix, Λ::ReMat)

Overwrite `L` with `Λ'LΛ + I`
"""
function scaleinflate! end

function scaleinflate!(Ljj::Diagonal{T}, Λj::ReMat{T,1}) where {T}
    Ljjd = Ljj.diag
    broadcast!((x, λsqr) -> x * λsqr + 1, Ljjd, Ljjd, abs2(first(Λj.λ)))
    Ljj
end

function scaleinflate!(Ljj::Matrix{T}, Λj::ReMat{T,1}) where {T}
    lambsq = abs2(only(Λj.λ))
    @inbounds for i in diagind(Ljj)
        Ljj[i] *= lambsq
        Ljj[i] += one(T)
    end
    Ljj
end

function scaleinflate!(Ljj::UniformBlockDiagonal{T}, Λj::ReMat{T,S}) where {T,S}
    A = Ljj.data
    m, n, l = size(A)
    m == n == S || throw(DimensionMismatch())
    λ = Λj.λ
    for f in 1:l
        lmul!(λ', rmul!(view(A, :, :, f), λ))
        for k in 1:S
            A[k, k, f] += one(T)
        end
    end
    Ljj
end

function scaleinflate!(Ljj::Matrix{T}, Λj::ReMat{T,S}) where{T,S}
    n = checksquare(Ljj)
    q, r = divrem(n, S)
    iszero(r) || throw(DimensionMismatch("size(Ljj, 1) is not a multiple of S"))
    λ = Λj.λ
    offset = 0
    @inbounds for k in 1:q
        inds = (offset + 1):(offset + S)
        lmul!(λ', rmul!(view(Ljj, inds, inds), λ))
        offset += S
    end
    for k in diagind(Ljj)
        Ljj[k] += 1
    end
    Ljj
end

"""
    skewscale!(Ljj, re::ReMat, ind::Integer)

Overwrite `Ljj` by `symmetrize!(λdot(re, ind)'Ljj*re.λ)`
"""
function skewscale! end

function skewscale!(Ljj::Diagonal{T}, re::ReMat{T,1}, ind::Integer) where {T}
    @assert isone(ind)
    Ljj.diag .*= (first(re.λ.data) * T(2))
    Ljj
end

function skewscale!(Ljj::Matrix{T}, re::ReMat{T,1}, ind::Integer) where {T}
    @assert isone(ind)
    Ljj .*= first(re.λ.data)
    symmetrize!(Ljj)
end

function skewscale!(Ljj::UniformBlockDiagonal{T}, re::ReMat{T,S}, ind::Integer) where {T,S}
    λd = λdot(re, ind)
    λ = re.λ
    Ljjd = Ljj.data
    for k in axes(Ljjd, 3)
        symmetrize!(lmul!(λd', rmul!(view(Ljjd, :, :, k), λ)))
    end
    Ljj
end

"""
updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.reterms` (used for λ only)

This is the crucial step in evaluating the objective, for a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A = m.A
    L = m.L
    k = length(m.allterms)
    for j = 1:k                         # copy lower triangle of A to L
        for i = j:k
            copyto!(L[Block(i, j)], A[Block(i, j)])
        end
    end
    for (j, cj) in enumerate(m.reterms)
        scaleinflate!(L[Block(j, j)], cj)  # pre- and post-multiply by Λ' and Λ, add I
        for i = (j+1):k         # postmultiply column by Λ
            rmulΛ!(L[Block(i, j)], cj)
        end
        for jj = 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', L[Block(j, jj)])
        end
    end
    for j = 1:k                         # blocked Cholesky
        Ljj = L[Block(j, j)]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        for jj = 1:(j-1)
            rankUpdate!(LjjH, L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i = (j+1):k
            Lij = L[Block(i, j)]
            for jj = 1:(j-1)
                mul!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    m
end

"""
updateLdot!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, and the sensitivites, `m.Ldot`, from `m.A` and `m.reterms`

This is the crucial step in evaluating the objective and gradient for a new parameter value.
"""
function updateLdot!(m::LinearMixedModel{T}) where {T}
    A = m.A
    L = m.L
    Ldot = m.Ldot
    k = length(m.allterms)
    for j = 1:k             # copy lower triangle of A to L and to each element of Ldot
        for i = j:k
            copyto!(L[Block(i, j)], A[Block(i, j)])
            for L̇ in Ldot
                copyto!(L̇[Block(i, j)], A[Block(i, j)])
            end
        end
    end
    offset = 0
    for (j, cj) in enumerate(m.reterms)
        scaleinflate!(L[Block(j, j)], cj)
        for (jj, kk) in enumerate(cj.inds)
            skewscale!(Ldot[offset + jj][Block(j, j)], cj, kk)
        end
        for i = (j+1):k         # postmultiply column by Λ
            rmulΛ!(L[Block(i, j)], cj)
        end
        for jj = 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', L[Block(j, jj)])
        end
    end
    for j = 1:k                         # blocked Cholesky
        Ljj = L[Block(j, j)]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        for jj = 1:(j-1)
            rankUpdate!(LjjH, L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i = (j+1):k
            Lij = L[Block(i, j)]
            for jj = 1:(j-1)
                mul!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    m
end
