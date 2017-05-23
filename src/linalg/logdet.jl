"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD{T<:Number}(d::Diagonal{T}) = sum(log, d.diag)

function LD{T}(d::Diagonal{LowerTriangular{T, Matrix{T}}})
    s = log(one(T))
    for dd in d.diag, i in diagind(dd)
        s += log(dd[i])
    end
    s
end

LD(d::DenseMatrix) = sum(i -> log(d[i]), diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I))` evaluated in place.
"""
function logdet{T}(m::LinearMixedModel{T})
    blks = m.L.data.blocks
    s = zero(T)
    for (k, t) in zip(diagind(blks), m.trms)
        if !isa(t, MatrixTerm)
            s += LD(blks[k])
        end
    end
    2.*s
end

logdet{T}(m::GeneralizedLinearMixedModel{T}) = logdet(m.LMM)
