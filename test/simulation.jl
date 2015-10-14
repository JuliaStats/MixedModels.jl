using DataFrames,DataArrays,Distributions,StatsBase,MixedModels

"""
Create a DataFrame with crossed grouping factors, `S` and `I`
with `M` and `N` levels, respectively.  The condition, `C`,
should be a vector of length 2.  The effect of all the `rep` calls
is similar to `expand.grid` in R.
"""
function mkdata(C::Vector=[zero(Int8),one(Int8)],N=10,M=100)
    l = length(C)
    DataFrame(y = zeros(l*N*M),
              C = rep(C,N*M),
              I = rep(rep(compact(pool(collect(1:N))),fill(l,N)),M),
              S = rep(compact(pool(collect(1:M))),fill(l*N,M)))
end

"""
Simulate a single response vector for a linear mixed model with
fixed-effects parameter β, and independent random effects for intercept
and condition, `C`, by subject and by item.  The function modifies the
first argument, `dat`, replacing dat[:y].
"""
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

const dat = mkdata();
const Xs = hcat(ones(2000),dat[:C]);
const Xst = Xs'
srand(1234321);


sim1!(dat,[2000.,0.]);
ms1 = fit!(lmm(y ~ 1+C + (1+C|S) + (1+C|I),dat));
#ms2 = fit!(lmm(y ~ 1+C + (1+C|S) + (1|I),dat))
