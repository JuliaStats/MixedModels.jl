using Chairmarks, MixedModels, StandardizedPredictors
using MixedModels: dataset, FormulaTerm, Table

@isdefined(contrasts) || const contrasts = Dict{Symbol, Any}()

contrasts[:F] = HelmertCoding()       # mrk17_exp1
contrasts[:P] = HelmertCoding()       # mrk17_exp1
contrasts[:Q] = HelmertCoding()       # mrk17_exp1
contrasts[:lQ] = HelmertCoding()      # mrk17_exp1
contrasts[:lT] = HelmertCoding()      # mrk17_exp1
contrasts[:ch] = HelmertCoding()      # contra
contrasts[:load] = HelmertCoding()    # kb07
contrasts[:prec] = HelmertCoding()    # kb07
contrasts[:service] = HelmertCoding() # insteval
contrasts[:spkr] = HelmertCoding()    # kb07
contrasts[:height] = Center()         # grouseticks
contrasts[:gender] = HelmertCoding()  # verbagg
contrasts[:btype] = EffectsCoding()   # verbagg
contrasts[:situ] = HelmertCoding()    # verbagg
contrasts[:mode] = HelmertCoding()    # verbagg

tbl = Table(
    dsnm = [
        :dyestuff2, :dyestuff, :pastes, :pastes, :machines, :penicillin,
        :sleepstudy, :sleepstudy, :sleepstudy, :sleepstudy, :kb07, :kb07, 
        :mrk17_exp1, :kb07, :insteval, :insteval, :mrk17_exp1, :d3, :ml1m,
    ],
    secs = append!(
        fill(0.1f0, 12),
        fill(1.0f0, 2),
        fill(5.0f0, 2),
        fill(25.0f0, 4)
    ),
    frm = FormulaTerm[
        @formula(yield ~ 1 + (1|batch)),
        @formula(yield ~ 1 + (1|batch)),
        @formula(strength ~ 1 + (1 | batch & cask)),
        @formula(strength ~ 1 + (1 | batch / cask)),
        @formula(score ~ 1 + (1 | Worker) + (1 | Machine)),
        @formula(diameter ~ 1 + (1 | plate) + (1 | sample)),
        @formula(reaction ~ 1 + days + (1 | subj)),
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        @formula(reaction ~ 1 + days + (1 | subj) + (0 + days | subj)),
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(log(rt_trunc) ~ 1 + spkr + prec + load + (1 | subj) + (1 | item)),
        @formula(log(rt_trunc) ~ 1 + spkr * prec * load + (1 | subj) + (1 + prec | item)),
        @formula(1000 / rt ~ 1 + F * P * Q * lQ * lT + (1 | item) + (1 | subj)),
        @formula(
            log(rt_trunc) ~
                1 + spkr * prec * load + (1 + spkr + prec + load | subj) +
                (1 + spkr + prec + load | item)
        ),
        @formula(y ~ 1 + service * dept + (1 | s) + (1 | d)),
        @formula(y ~ 1 + service + (1 | s) + (1 | d) + (1 | dept)),
        @formula(
            1000 / rt ~
                1 + F * P * Q * lQ * lT + (1 + P + Q + lQ + lT | item) +
                (1 + F + P + Q + lQ + lT | subj)
        ),
        @formula(y ~ 1 + u + (1 + u | g) + (1 + u | h) + (1 + u | i)),
        @formula(y ~ 1 + (1 | g) + (1 | h)),
    ]
)

# linmark(f, d, t) = @b fit(MixedModel, f, dataset(d); contrasts, progress=false) seconds=t

@track (@b (first(tbl.frm), dataset(first(tbl.dsnm))) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[2], dataset(tbl.dsnm[2])) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[3], dataset(tbl.dsnm[3])) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[4], dataset(tbl.dsnm[4])) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[5], dataset(tbl.dsnm[5])) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[6], dataset(tbl.dsnm[6])) fit(MixedModel, first(_), last(_); progress=false)).time
@track (@b (tbl.frm[7], dataset(tbl.dsnm[7])) fit(MixedModel, first(_), last(_); progress=false)).time

# function runbmrk(tbl)
#     return Table([(; bmk=linmark(f, d, t), dsnm = d, frm=f) for (d, t, f) in tbl])
# end

# gltbl = Table(
#     dsnm = [:contra, :contra, :verbagg, :grouseticks],
#     secs = [2.0, 2.0, 15.0, 15.0],
#     dist = [Bernoulli(), Bernoulli(), Bernoulli(), Poisson()],
#     frm = FormulaTerm[
#         @formula(use ~ 1+age+abs2(age)+urban+livch+(1|urban&dist)),
#         @formula(use ~ 1+age+abs2(age)+urban+(≠("0"))(livch)+(1+urban|dist)),
#         @formula(r2 ~ 1+anger+gender+btype+situ+(1|subj)+(1|item)),
#         @formula(ticks ~ 1+year+height+(1|index)+(1|brood)+(1|location)),
#     ]
# )

# glmark(f, d, r, t; init_from_lmm=()) = @b fit(MixedModel, f, dataset(d), r; init_from_lmm, contrasts, progress=false) seconds=t

# function runglbmk(tbl; init_from_lmm=())
#     return Table((; bmk=glmark(f, d, r, t; init_from_lmm), dsnm=d, dist=r, frm=f) for (d, t, r, f) in tbl)
# end
