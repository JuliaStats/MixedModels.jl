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
    @inbounds for j in 1:k, i in 1:m
        s += log(dat[i,i,j])
    end
    s
end

function LD(d::DenseMatrix{T}) where {T}
    s = log(one(T))
    for i in 1:LinearAlgebra.checksquare(d)
        s += log(d[i, i])
    end
    s
end

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I)) + log(det(LX*LX'))` evaluated in place.

Here LX is the diagonal term corresponding to the fixed-effects in the blocked
lower Cholesky factor.
"""
function LinearAlgebra.logdet(m::LinearMixedModel{T}) where {T}
    s = log(one(T))
    Ldat = m.L.data
    @inbounds for (i, trm) in enumerate(m.trms)
        isa(trm, AbstractFactorReTerm) && (s += LD(Ldat[Block(i, i)]))
    end
    if m.optsum.REML
        feindex = length(m.trms) - 1
        fetrm = m.trms[feindex]
        if isa(fetrm, MatrixTerm)
            lblk = Ldat[Block(feindex, feindex)]
            for i in 1:fetrm.rank
                s += log(lblk[i, i])
            end
        end
    end
    2s
end

LinearAlgebra.logdet(m::GeneralizedLinearMixedModel) = logdet(m.LMM)
