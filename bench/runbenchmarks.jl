using Chairmarks, MixedModels, StatsModels, TypedTables
using MixedModels: dataset

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

tbl = Table(
    dsnm = [
        :dyestuff2, :dyestuff, :machines, :pastes, :pastes, :penicillin,
        :sleepstudy, :sleepstudy, :sleepstudy, :sleepstudy, :kb07, :kb07, 
        :mrk17_exp1, :insteval, :insteval, :kb07, :mrk17_exp1, :d3, :ml1m,
    ],
    secs = append!(
        fill(0.1f0, 12),
        [1.0f0],
        fill(5.0f0, 3),
        fill(25.0f0, 4)
    ),
    frm = StatsModels.FormulaTerm[
        @formula(yield ~ 1 + (1|batch)),
        @formula(yield ~ 1 + (1|batch)),
        @formula(score ~ 1 + (1 | Worker) + (1 | Machine)),
        @formula(strength ~ 1 + (1 | batch & cask)),
        @formula(strength ~ 1 + (1 | batch / cask)),
        @formula(diameter ~ 1 + (1 | plate) + (1 | sample)),
        @formula(reaction ~ 1 + days + (1 | subj)),
        @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
        @formula(reaction ~ 1 + days + (1 | subj) + (0 + days | subj)),
        @formula(reaction ~ 1 + days + (1 + days | subj)),
        @formula(log(rt_trunc) ~ 1 + spkr + prec + load + (1 | subj) + (1 | item)),
        @formula(log(rt_trunc) ~ 1 + spkr * prec * load + (1 | subj) + (1 + prec | item)),
        @formula(1000 / rt ~ 1 + F * P * Q * lQ * lT + (1 | item) + (1 | subj)),
        @formula(y ~ 1 + service * dept + (1 | s) + (1 | d)),
        @formula(y ~ 1 + service + (1 | s) + (1 | d) + (1 | dept)),
        @formula(
            log(rt_trunc) ~
                1 + spkr * prec * load + (1 + spkr + prec + load | subj) +
                (1 + spkr + prec + load | item)
        ),
        @formula(
            1000 / rt ~
                1 + F * P * Q * lQ * lT + (1 + P + Q + lQ + lT | item) +
                (1 + F + P + Q + lQ + lT | subj)
        ),
        @formula(y ~ 1 + u + (1 + u | g) + (1 + u | h) + (1 + u | i)),
        @formula(y ~ 1 + (1 | g) + (1 | h)),
    ]
)

linmark(f, d, t) = @b fit(MixedModel, f, dataset(d); contrasts, progress=false) seconds=t

function runbmrk(tbl)
    Table([(; bmk=linmark(f, d, t), dsnm = d, frm=f) for (d, t, f) in tbl])
end
