const libpastix = "/var/tmp/pastix_release_351ef60/install/libpastix.so"

function chkMtx(m::SparseMatrixCSC{Float64,Int32})
    cp = [convert(Ptr{Cint},m.colptr)]
    rv = [convert(Ptr{Cint},m.rowval)]
    xv = [convert(Ptr{Cdouble},m.nzval)]
    ccall((:pastix_checkMatrix,libpastix),Void,
          (Cint,Cint,Cint,Cint,Cint,
           Ptr{Ptr{Cint}},Ptr{Ptr{Cint}},Ptr{Ptr{Cdouble}},Ptr{Ptr{Cint}},Cint),
          0,2,0,1,m.n,cp,rv,xv,convert(Ptr{Ptr{Cint}},0),1)
    cp, rv, xv
end

type PaStiX
    m::SparseMatrixCSC{Float64,Int32}
    perm::Vector{Int32}
    iperm::Vector{Int32}
    iparm::Vector{Int32}
    dparm::Vector{Float64}
    pastixData::Vector{Ptr{Void}}
    function PaStiX(m::SparseMatrixCSC{Float64,Int32})
        n = m.n
        m.m == n || error("m should be square and size(m) = $(size(m))")
        iparm = Array(Cint,128)
        dparm = Array(Cdouble,64)
        ccall((:pastix_initParam,libpastix),Void,(Ptr{Cint},Ptr{Cdouble}),iparm,dparm)
        new(m,Array(Int32,n),Array(Int32,n),iparm,dparm,[C_NULL])
    end
end

function pastix(p::PaStiX)
    m = p.m
    ccall((:pastix,libpastix),Void,
          (Ptr{Ptr{Void}},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cdouble},
           Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Cint,Ptr{Cint},Ptr{Cdouble}),
          p.pastixData,0,m.n,m.colptr,m.rowval,m.nzval,p.perm,p.iperm,C_NULL,0,p.iparm,p.dparm)
    p
end

type PaStiXn <: LinearMixedModel
    L::PaStiX
    Lambda::Diagonal
    RX::Base.LinAlg.Cholesky{Float64}
    X::ModelMatrix{Float64}             # fixed-effects model matrix
    ZXty::Vector{Float64}
    Zt::SparseMatrixCSC{Float64,Int32}
    ZXtZX::SparseMatrixCSC{Float64,Int32}
    Zty::Vector{Float64}
    beta::Vector{Float64}
    fnms::Vector
    lambda::Vector{Float64}
    mu::Vector{Float64}
    offsets::Vector
    u::Vector{Vector{Float64}}
    y::Vector
    REML::Bool
    fit::Bool
end

function lmmp(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); y = model_response(mf)
    reinfo = {retrm(r,mf.df) for r in
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
    Zt = SparseMatrixCSC(offsets[end],n,colptr,rv,vec(hcat(Xs...)'))
    Zt
end
