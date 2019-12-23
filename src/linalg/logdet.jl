"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::UniformBlockDiagonal{T}) where {T}
    dat = d.data
    m, n, k = size(dat)
    m == n || throw(ArgumentError("Blocks of d must be square"))
    s = log(one(T))
    @inbounds for j = 1:k, i = 1:m
        s += log(dat[i, i, j])
    end
    s
end

function LD(d::DenseMatrix{T}) where {T}
    s = log(one(T))
    for i = 1:LinearAlgebra.checksquare(d)
        s += log(d[i, i])
    end
    s
end

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I)) + m.optsum.REML * log(det(LX*LX'))`
evaluated in place.

Here LX is the diagonal term corresponding to the fixed-effects in the blocked
lower Cholesky factor.
"""
function LinearAlgebra.logdet(m::LinearMixedModel{T}) where {T}
    s = log(one(T))
    L = m.L
    nre = length(m.reterms)
    @inbounds for i = 1:nre
        s += LD(L[Block(i, i)])
    end
    if m.optsum.REML
        feindex = nre + 1
        s += LD(L[Block(feindex, feindex)])
    end
    2s
end
