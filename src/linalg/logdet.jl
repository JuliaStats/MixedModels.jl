"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::UniformBlockDiagonal{T}) where T
    dat = d.data
    m, n, k = size(dat)
    m == n || throw(ArgumentError("Blocks of d must be square"))
    s = log(one(T))
    @inbounds for j in 1:k, i in 1:m
        s += log(dat[i,i,j])
    end
    s
end

function LD(d::DenseMatrix{T}) where T
    s = log(one(T))
    for i in 1:Base.LinAlg.checksquare(d)
        s += log(d[i, i])
    end
    s
end

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` evaluated in place.
"""
function logdet(m::LinearMixedModel{T}) where {T}
    s = log(one(T))
    Ldat = m.L.data
    for (i, trm) in enumerate(m.trms)
        if isa(trm, AbstractFactorReTerm)
            s += LD(m.L.data[Block(i, i)])
        end
    end
    2s
end

logdet(m::GeneralizedLinearMixedModel) = logdet(m.LMM)
