## Utilities

## Create a Base.Cholesky object from a StridedMatrix
function cholesky(A::StridedMatrix{Float64},uplo::Symbol)
    Base.LinAlg.chksquare(A)
    Base.Cholesky{Float64,typeof(A)}(A,string(uplo)[1])
end

typealias TRI Base.LinAlg.AbstractTriangular

typealias CHMfac CHOLMOD.Factor
typealias CHMsp CHOLMOD.Sparse

LinAlg.cholfact(A,beta,ll::Bool) = cholfact(A,beta)

const Sym = CHOLMOD.SYM
chm_scale!(A,S,typ) = CHOLMOD.scale!(Dense(S),typ,A)

chfac(ch::Base.Cholesky) = ch.factors
cholf!(L,A,b) = CHOLMOD.update!(L,A,b)
perm(L::CHMfac) = Base.SparseMatrix.CHOLMOD.get_perm(L)
cholesky{T,S}(A::UpperTriangular{T,S}) = Base.Cholesky(A.data,:U)
cholesky{T,S}(A::LowerTriangular{T,S}) = Base.Cholesky(A.data,:L)
ltri(m::StridedMatrix) = LowerTriangular(m)

"Version of cholfact of a symmetric matrix that works in both 0.3 and 0.4 versions"
Base.cholfact(s::Symmetric{Float64}) = cholfact(symcontents(s), symbol(s.uplo))

"Create a cross-tabulation of two `PooledDataVector` objects"
crosstab(a::PooledDataVector,b::PooledDataVector) =
    counts(a.refs,b.refs,(length(a.pool),length(b.pool)))

## grplevels(m) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(v::Vector) = [length(isa(f,PooledDataVector) ? f.pool : unique(f)) for f in v]
grplevels(dd::DataFrame,nms::Vector) = grplevels([getindex(dd,nm) for nm in nms])

"Check whether the levels in factors or arrays are nested"
isnested(v::Vector) = length(v) == 1 || length(Set(zip(v...))) == maximum(grplevels(v))
isnested(dd::DataFrame,nms::Vector) = isnested([getindex(dd,nm) for nm in nms])

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
## FIXME: Change this behavior in DataFrames/src/statsmodels/formula.jl
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

"Scale b by the componentwise inverse of sc"
function scaleinv!(b::StridedVector,sc::StridedVector)
    (n = length(b)) == length(sc) || throw(DimensionMismatch(""))
    @inbounds for i in 1:n
        b[i] /= sc[i]
    end
    b
end

"Return the contents (one triangle only) of the symmetric matrix s for versions 0.3 and 0.4"
symcontents(s::Symmetric) = VERSION ≥ v"0.4-" ? s.data : s.S

"Read a sample data set from the data directory"
function rdata(nm::Union{AbstractString,Symbol})
    DataFrame(read_rda(Pkg.dir("MixedModels","data",string(nm,".rda")))[string(nm)])
end

"Return a block in the Zt matrix from the transposed model matrix and grouping factor"
function ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = Int(maximum(v))
    NR = nr*nblk                        # number of rows in result
    colptr = cumsum(vcat(1,fill(nr,nc)))
    rowval = vec(reshape([1:NR;],(nr,nblk))[:,v])
    if length(m) < typemax(Int32)
        colptr = convert(Vector{Int32},colptr)
        rowval = convert(Vector{Int32},rowval)
    end
    SparseMatrixCSC(NR,nc,colptr,rowval,vec(m))
end
ztblk(m::Matrix,v::PooledDataVector) = ztblk(m,v.refs)

"A set of indices determined by p and l - used as an iterator"
immutable inds
    p::Int
    l::Int
    function inds(p,l)
        p = Int(p)
        l = Int(l)
        p > 0 && l > 0 || error("p = $p and l = $l must both be positive")
        new(p,l)
    end
end

Base.length(i::inds) = i.l
Base.start(i::inds) = 0
Base.next(i::inds,j::Int) = (j*i.p + (1:i.p), j+1)
Base.done(i::inds,j::Int) = j ≥ i.l
