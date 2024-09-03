"""
    LD(A::Diagonal)
    LD(A::HBlikDiag)
    LD(A::DenseMatrix)

Return `log(det(tril(A)))` evaluated in place.
"""
LD(d::Diagonal{T}) where {T<:Number} = sum(log, d.diag)

function LD(d::UniformBlockDiagonal{T}) where {T}
    dat = d.data
    return sum(log, dat[j, j, k] for j in axes(dat, 2), k in axes(dat, 3))
end

LD(d::DenseMatrix{T}) where {T} = @inbounds sum(log, d[k] for k in diagind(d))

"""
    logdet(m::LinearMixedModel)

Return the value of `log(det(Λ'Z'ZΛ + I)) + m.optsum.REML * log(det(LX*LX'))`
evaluated in place.

Here LX is the diagonal term corresponding to the fixed-effects in the blocked
lower Cholesky factor.
"""
function LinearAlgebra.logdet(m::LinearMixedModel{T}) where {T}
    L = m.L
    @inbounds s = sum(j -> LD(L[kp1choose2(j)])::T, axes(m.reterms, 1))
    if m.optsum.REML
        lastL = last(L)::Matrix{T}
        s += LD(lastL)        # this includes the log of sqrtpwrss
        s -= log(last(lastL)) # so we need to subtract it from the sum
    end
    return (s + s)::T  # multiply by 2 b/c the desired det is of the symmetric mat, not the factor
end
