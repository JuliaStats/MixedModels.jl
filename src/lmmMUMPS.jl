type lmmMUMPS
    dm::DMumps
    A::SparseMatrixCSC{Float64,Int32}
    ZXt::SparseMatrixCSC{Float64,Int32}
    offsets::Vector{Int}
    a::Vector{Float64}
    jcn::Vector{Int32}
    y::Vector{Float64}
    mu::Vector{Float64}
    res::Vector{Float64}
end

function expandcptr(S::SparseMatrixCSC)
    cp = S.colptr
    J = similar(S.rowval)
    for j in 1:S.n
        fill!(sub(J,cp[j]:cp[j+1]-1),j)
    end
    J
end

function Ztblk(m::Matrix,v::Vector)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nc*nblk                        # number of rows in result
    SparseMatrixCSC(NR,nr,
                    cumsum(vcat(1,fill(nc,(nr,)))),           # colptr
                    vec(reshape([1:NR],(nc,int(nblk)))[:,v]), # rowval
                    vec(m'))                                  # nzval
end

function rvblk(m,v::Vector)
    nblk = maximum(v)
    vec(reshape(Int32[1:m*nblk],(m,int(nblk)))[:,v])
end

Zt(fm::LMMGeneral) = vcat([Ztblk(m,v) for (m,v) in zip(fm.Xs,fm.inds)]...)
    
function lmmMUMPS(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); n,p = size(X); y = model_response(mf)
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    (k = length(retrms)) > 0 || error("Formula $f has no random-effects terms")
    k == 1 && return MixedModels.retrm(retrms[1])
    reinfo = [MixedModels.retrm(t,mf.df) for t in retrms]
    Xs = Matrix{Float64}[r[1] for r in reinfo]
    facs = [r[2] for r in reinfo]
    fnms = String[r[3] for r in reinfo]
    pvec = [size(m,2) for m in Xs]
    refs = map(f->getfield(f,:refs),facs)
    levs = map(f->getfield(f,:pool),facs)
    nlev = Int[length(l) for l in levs]
    offsets = [0, cumsum(nlev .* pvec)]
    q = offsets[end]
    Ti = q < typemax(Int32) ? Int32 : Int64
    rv = vec(vcat([convert(Matrix{Ti},
                           reshape([(offsets[i]+1):offsets[i+1]],
                                   (pvec[i],nlev[i]))[:,refs[i]]) for i in 1:k]...))
    np = sum(pvec)
    colptr = convert(Vector{Ti},[1:np:(np*n + 1)])
    ZXt = vcat(SparseMatrixCSC(offsets[end],n,colptr,rv,vec(hcat(Xs...)')),
               convert(SparseMatrixCSC{Float64,Int32},X.m'))
    ZXtZX = ZXt * ZXt' + speye(ZXt.m)
    ZXty = ZXt * y
    dm = DMumps()
    iv = [getfield(dm.icntl,nm)::Cint for nm in names(dm.icntl)]
#    iv[1] = iv[2] = iv[3] = -1; iv[4] = 0
    iv[33] = 1
    dm.icntl = Icntl(iv...)
    dm.sym = 1
    dm.lrhs = dm.n = ZXtZX.m
    dm.nz = nfilled(ZXtZX)
    dm.irn = pointer(ZXtZX.rowval)
    jcn = expandcptr(ZXtZX)
    dm.jcn = pointer(jcn)
    a = copy(ZXtZX.nzval)
    dm.a = pointer(a)
    dm.job = 6
    dm.rhs = pointer(ZXty)
    dm.nrhs = 1
    ccall((:dmumps_c,"libdmumps_seq"),Void,(Ptr{DMumps},),&dm)
    lmmMUMPS(dm,ZXtZX,ZXt,offsets,a,jcn,y)
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
