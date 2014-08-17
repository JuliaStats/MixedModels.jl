## Utilities

## Return a block in the Zt matrix from one term.
function ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nr*nblk                        # number of rows in result
    cf = length(m) < typemax(Int32) ? int32 : int64 # conversion function
    SparseMatrixCSC(NR,nc,
                    cf(cumsum(vcat(1,fill(nr,(nc,))))), # colptr
                    cf(vec(reshape([1:NR],(nr,int(nblk)))[:,v])), # rowval
                    vec(m))            # nzval
end
ztblk(m::Matrix,v::PooledDataVector) = ztblk(m,v.refs)

function Base.cholfact(ch::Base.LinAlg.Cholesky,inds::UnitRange{Int})
    if VERSION.minor < 4
        return Base.LinAlg.Cholesky(ch[symbol(ch.uplo)].data[inds,inds],ch.uplo)
    end
    tr = ch[:UL]
    dd = tr.data[inds,inds]
    Base.LinAlg.Cholesky{eltype(dd),typeof(dd),istril(tr)?(:L):(:U)}(dd)
end
