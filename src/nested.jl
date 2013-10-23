type LMMNested <: LinearMixedModel
    Zt::SparseMatrixCSC
    lambda::Vector
    RX::Cholesky
    X::ModelMatrix                      # fixed-effects model matrix
    Xty::Vector
    ZtX::Matrix
    ZtZ::SparseMatrixCSC
    L::SparseMatrixCSC
    Zty::Vector
    beta::Vector
    fnms::Vector
    mu::Vector
    offsets::Vector
    perm::Vector                        # etree permutation
    pinv::Vector                        # inverse permutation
    u::Vector{Vector}
    y::Vector
    REML::Bool
    fit::Bool
end

function LMMNested(X::ModelMatrix,Xs::Vector,refs::Vector,levs::Vector,y::Vector,
                   fnms::Vector,pvec::Vector,nlev::Vector,offsets::Vector)
    all(pvec .== 1) || error("Nested non-scalar random effects not yet available")
    n,p = size(X); k = length(Xs); q = offsets[end]; mx = max(q,n*k+1)
    Ti = mx < typemax(Uint8) ? Uint8 :
    (mx < typemax(Uint16) ? Uint16 :
     (mx < typemax(Uint32) ? Uint32 : Uint64))
    Zt = SparseMatrixCSC(q,n,convert(Vector{Ti},[1:k:(k*n + 1)]),
                         convert(Vector{Ti},vec(broadcast(+,hcat(refs...)',
                                                          offsets[1:k]))),
                         vec(hcat([x[:,1] for x in Xs]...)'))
    ZtZ = triu(Zt * Zt')
    tr,perm = etree(ZtZ,true)
    pinv = invperm(perm)
    ZtZ = Base.SparseMatrix.csc_symperm(ZtZ,pinv) # post-order ZtZ
    roots = pinv[[1:q][tr .== 0]] # nodes that are roots of the elimination tree
    L = ZtZ'
end

## Special-purpose Cholesky decomposition for the sparse crossproduct from nested factors
function cholnested{Tv,Ti}(L::SparseMatrixCSC{Tv,Ti},C::SparseMatrixCSC{Tv,Ti},beta=zero(Tv))
    (n = A.n) == A.m == L.n == L.m || error("Dimension mismatch")
    istriu(A) && istril(L) ||
        error("A must be symmetric stored in upper triangle and L lower triangular")
    Ap = A.colptr; Ai = A.rowval; Ax = A.nzval; Lp = L.colptr; Li = L.rowval; Lx = L.nzval
    x = Array(Tv,n)                     # workspace
    for k in 1:n                        # copy A' to L, inflating the diagonal
        Lx[Lp[k]] = Ax[Ap[k+1]-1] + beta
        for p in (Lp[k]+1):(Lp[k+1]-1)
            j = Li[p]                   # row of L == column of A
            for q in Ap[j]:(Ap[j+1]-1)  # (should be able to short-circuit this by moving pointers)
                Ai[q] == k && (Lx[p] = Ax[q])
            end
        end
    end
    L
end

