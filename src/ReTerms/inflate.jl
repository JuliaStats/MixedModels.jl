"""
Equivalent to `A += I` without making a copy of A
"""
function inflate!(A::HBlkDiag)
    Aa = A.arr
    r,s,k = size(Aa)
    for j in 1:k, i in 1:min(r,s)
        Aa[i,i,j] += 1
    end
    A
end

inflate!(D::Diagonal{Float64}) = (d = D.diag; for i in eachindex(d) d[i] += 1 end; D)

function inflate!{T<:AbstractFloat}(A::StridedMatrix{T})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        @inbounds A[i,i] += 1
    end
    A
end
