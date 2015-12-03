"""
`densify(S[,threshold])`

Convert sparse `S` to `Diagonal` if S is diagonal
Convert sparse `S` to dense if the proportion of nonzeros exceeds `threshold`.
A no-op for other matrix types.
"""
function densify(S,threshold=0.3)
    if !issparse(S)  # non-sparse matrices are not modified
        return S
    end
    m,n = size(S)
    if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
        return Diagonal(diag(S))
    end
    if nnz(S)/(*(size(S)...)) ≤ threshold # very sparse matrices left as is
        return S
    end
    if isbits(eltype(S))
        return full(S)
    end
    nzs = nonzeros(S)
    nz1 = nzs[1]
    T = typeof(nz1)
    if !isa(nz1,Array) || !isbits(eltype(nz1)) # branch not tested
        error("Nonzeros must be a bitstype or an array of same")
    end
    sz1 = size(nz1)
    if any(x->typeof(x) ≠ T || size(x) ≠ sz1, nzs) # branch not tested
        error("Inconsistent dimensions or types in array nonzeros")
    end
    M,N = size(S)
    m,n = size(nz1,1),size(nz1,2) # this construction allows for nz1 to be a vector
    res = Array(eltype(nz1),M*m,N*n)
    rv = rowvals(S)
    for j in 1:size(S,2)
        for k in nzrange(S,j)
            copy!(sub(res,(rv[k]-1)*m+(1:m),(j-1)*n+(1:n)),nzs[k])
        end
    end
    res
end
