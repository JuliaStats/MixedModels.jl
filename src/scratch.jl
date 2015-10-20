using DataFrames,DataArrays,Distributions,StatsBase,MixedModels
include("/home/bates/.julia/v0.4/ReTerms/test/data.jl")
f = Reaction ~ 1+Days + (1+Days|Subject)
fr = slp
mf = ModelFrame(f,fr)
X = ModelMatrix(mf)
y = convert(Vector{Float64},DataFrames.model_response(mf))
mm = lmm(Reaction ~ 1+Days + (1+Days|Subject),slp)
mm[:θ] = [1.,0,1]
objective(mm)
                                    # process the random-effects terms
retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
length(retrms) > 0 || error("Formula $f has no random-effects terms")
re = [MixedModels.remat(e,mf.df) for e in retrms]
@show re
m = fit!(LinearMixedModel(re,X.m,y),true,:LN_BOBYQA)
function mkdata(C::Vector=[zero(Int8),one(Int8)],N=10,M=100)
    l = length(C)
    DataFrame(y = zeros(l*N*M),
              C = rep(C,N*M),
              I = rep(rep(compact(pool(collect(1:N))),fill(l,N)),M),
              S = rep(compact(pool(collect(1:M))),fill(l*N,M)))
end
const dat = mkdata()
function sim1!(dat::DataFrame,β::Vector,σ=600.,σRS=300.,σRI=30.)
    y,C,I,S = dat[:y],dat[:C],dat[:I],dat[:S]
    length(β) == length(unique(C)) == 2 || throw(DimensionMismatch())
    # vectors of random effects (subject-int,subject-C,item-int,item-C)
    M,N = length(S.pool),length(I.pool)
    re = Vector{Float64}[σRS*randn(M),σRS*randn(M)*0.5,
                         σRI*randn(N),σRI*randn(N)*0.5]
    I,S = I.refs,S.refs
    for ii in eachindex(y)
        yi = β[1] + re[1][S[ii]] + re[3][I[ii]] + σ*randn()
        if (cc = C[ii]) ≠ 0
            yi += cc*(β[2]+re[2][S[ii]]+re[4][I[ii]])
        end
        y[ii] = yi
    end
    dat
end
srand(1234321)
sim1!(dat,[2000.,0.]);
m4 = fit!(LinearMixedModel(y ~ 1+C+(1|S)+(1|I),dat),true,:LN_BOBYQA)
m4.opt
m3 = fit!(lmm(y ~ 1+C+(1|S)+(0+C|S)+(1|I),dat),true,:LN_BOBYQA)
m3.opt
m2 = fit!(lmm(y ~ 1+C+(1|S)+(0+C|S)+(1|I)+(0+C|I),dat),true,:LN_BOBYQA)
m2.opt
m1 = fit!(lmm(y ~ 1+C+(1+C|S)+(1+C|I),dat),true,:LN_BOBYQA)
m1.opt
ccall((:jl_cholmod_sizeof_long, :libsuitesparse_wrapper),Csize_t,())
