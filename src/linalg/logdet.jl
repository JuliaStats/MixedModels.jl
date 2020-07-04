"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::UniformBlockDiagonal{T}) where {T}
    dat = d.data
    sum(log, dat[j, j, k] for j in axes(dat, 2), k in axes(dat, 3))
end

LD(d::DenseMatrix{T}) where {T} = sum(log, d[k] for k in diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I)) + m.optsum.REML * log(det(LX*LX'))`
evaluated in place.

Here LX is the diagonal term corresponding to the fixed-effects in the blocked
lower Cholesky factor.
"""
function LinearAlgebra.logdet(m::LinearMixedModel{T}) where {T}
    L = m.L
    nre = m.dims.nretrms
    s = sum(LD, L[Block(j, j)] for j in Base.OneTo(nre))
    if m.optsum.REML
        feindex = nre + 1
        s += LD(L[Block(feindex, feindex)])
    end
    s + s
end
