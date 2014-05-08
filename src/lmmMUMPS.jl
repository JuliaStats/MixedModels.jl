type lmmMUMPS
    dm::DMumps
    A::SparseMatrixCSC{Float64,Int32}
    ZXt::SparseMatrixCSC{Float64,Int32}
    offsets::Vector{Int}
    a::Vector{Float64}
    jcn::Vector{Int32}
    y::Vector{Float64}
end

function lmmMUMPS(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); y = model_response(mf)
    reinfo = {MixedModels.retrm(r,mf.df) for r in
        filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)};
    (k = length(reinfo)) > 0 || error("Formula $f has no random-effects terms")
    Xs = Matrix{Float64}[r[1] for r in reinfo]; facs = [r[2] for r in reinfo];
    fnms = String[r[3] for r in reinfo]; pvec = Int[size(m,2) for m in Xs];
    refs = [f.refs for f in facs]; levs = [f.pool for f in facs]; n,p = size(X)
    nlev = [length(l) for l in levs]; offsets = [0, cumsum(nlev .* pvec)]
    q = offsets[end]; Ti = q < typemax(Int32) ? Int32 : Int64
    rv = vec(vcat([convert(Matrix{Ti},
                           reshape([(offsets[i]+1):offsets[i+1]],
                                   (pvec[i],nlev[i]))[:,refs[i]]) for i in 1:k]...))
    np = sum(pvec)
    colptr = convert(Vector{Ti},[1:np:(np*n + 1)])
    ZXt = vcat(SparseMatrixCSC(offsets[end],n,colptr,rv,vec(hcat(Xs...)')),
               convert(SparseMatrixCSC{Float64,Int32},X.m'))
    ZXtZX = ZXt * ZXt' 
    dm = DMumps()
    dm.sym = 1
    dm.n = ZXtZX.m
    dm.nz = nfilled(ZXtZX)
    dm.irn = pointer(ZXtZX.rowval)
    jcn = expandcptr(ZXtZX)
    dm.jcn = pointer(jcn)
    a = copy(ZXtZX.nzval)
    dm.a = pointer(a)
    dm.job = 1
    ccall((:dmumps_c,"libdmumps_seq"),Void,(Ptr{DMumps},),&dm)
    lmmMUMPS(dm,ZXtZX,ZXt,offsets,a,jcn,y)
end

function expandcptr(A::SparseMatrixCSC)
    res = similar(A.rowval)
    colpt = A.colptr
    for j in 1:A.n, k in colpt[j]:colpt[j+1]-1
        res[k] = j
    end
    res
end

function reml(lmm::lmmMUMPS,v::Vector{Float64})
    q = lmm.offsets[end]
    a = lmm.a
    irn = lmm.A.rowval
    ab = lmm.A.nzval
    lambda = ones(lmm.A.n)
    jcn = lmm.jcn
    @inbounds for k in 1:length(a)
        i = irn[k]
        j = jcn[k]
        a[k] = ab[k]*lambda[i]*lambda[j]
        i <= q && i == j && (a[k] += 1.) # inflate the diagonal
    end
    lmm.dm.job = 2
    ccall((:dmumps_c,"libdmumps_seq"),Void,(Ptr{DMumps},),&lmm.dm)
end
