## Utilities

function amalgamate1(Xs,p,λ)
    (k = length(λ)) == length(Xs) == length(p) || throw(DimensionMismatch(""))
    k == 1 && return (Xs,p,λ)
    if all([isa(ll,PDScalF) for ll in λ])
        return({vcat(Xs...)},[sum(p)],{PDDiagF(ones(length(λ)))},)
    end
    error("Composite code not yet written")
end

## amalgamate random-effects terms with identical grouping factors
function amalgamate(grps,Xs,p,λ,facs)
    np = Int[]; nXs = {}; nλ = {}; nfacs = {}
    ugrp = unique(grps)
    for u in ugrp
        inds = grps .== u
        (xv,pv,lv) = amalgamate1(Xs[inds],p[inds],λ[inds])
        append!(np, pv)
        append!(nXs,xv)
        append!(nλ,lv)
        append!(nfacs,{facs[inds[1]]})
    end
    ugrp,nXs,np,nλ,nfacs
end

## Version of cholfact of a symmetric matrix that works in both 0.3 and 0.4 series
Base.cholfact(s::Symmetric{Float64}) = cholfact(symcontents(s), symbol(s.uplo))

crosstab(a::PooledDataVector,b::PooledDataVector) =
    counts(a.refs,b.refs,(length(a.pool),length(b.pool)))

## grplevels(m) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(v::Vector) = [length(isa(f,PooledDataVector) ? f.pool : unique(f)) for f in v]
grplevels(dd::DataFrame,nms::Vector) = grplevels({getindex(dd,nm) for nm in nms})

## Check if the levels in factors or arrays are nested
isnested(v::Vector) = length(v) == 1 || length(Set(zip(v...))) == maximum(grplevels(v))
isnested(dd::DataFrame,nms::Vector) = isnested({getindex(dd,nm) for nm in nms})

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
## FIXME: Change this behavior in DataFrames/src/statsmodels/formula.jl
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

## Scale b by the componentwise inverse of sc
function scaleinv!(b::StridedVector,sc::StridedVector)
    (n = length(b)) == length(sc) || throw(DimensionMismatch(""))
    @inbounds for i in 1:n
        b[i] /= sc[i]
    end
    b
end

## Determine the contents of the symmetric matrix s for versions 0.3 and 0.4
symcontents(s::Symmetric) = VERSION ≥ v"0.4-" ? s.data : s.S

## Read a sample data set from the data directory.
function rdata(nm::Union(String,Symbol))
    DataFrame(read_rda(Pkg.dir("MixedModels","data",string(nm,".rda")))[string(nm)])
end

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


