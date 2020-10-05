using MixedModels
using MixedModels: dataset


@isdefined(fms) || const global fms = Dict(
    :dyestuff => [@formula(yield ~ 1 + (1|batch))],
    :dyestuff2 => [@formula(yield ~ 1 + (1|batch))],
    :d3 => [@formula(y ~ 1 + u + (1+u|g) + (1+u|h) + (1+u|i))],
    :insteval => [
        @formula(y ~ 1 + service + (1|s) + (1|d) + (1|dept)),
        @formula(y ~ 1 + service*dept + (1|s) + (1|d)),
    ],
    :kb07 => [
        @formula(rt_trunc ~ 1+spkr+prec+load+(1|subj)+(1|item)),
        @formula(rt_trunc ~ 1+spkr*prec*load+(1|subj)+(1+prec|item)),
        @formula(rt_trunc ~ 1+spkr*prec*load+(1+spkr+prec+load|subj)+(1+spkr+prec+load|item)),
    ],
    :pastes => [
        @formula(strength ~ 1 + (1|sample)),
        @formula(strength ~ 1 + (1|sample) + (1|batch)),
    ],
    :penicillin => [@formula(diameter ~ 1 + (1|plate) + (1|sample))],
    :sleepstudy => [
        @formula(reaction ~ 1 + days + (1|subj)),
        @formula(reaction ~ 1 + days + zerocorr(1+days|subj)),
        @formula(reaction ~ 1 + days + (1|subj) + (0+days|subj)),
        @formula(reaction ~ 1 + days + (1+days|subj)),
    ],
)

# for some reason it seems necessary to prime the pump in julia-1.6.0-DEV
@isdefined(fittedmodels) || const global fittedmodels = Dict{Symbol,Vector{LinearMixedModel}}(
    :dyestuff => [fit(MixedModel, only(fms[:dyestuff]), dataset(:dyestuff))]
);

function models(nm::Symbol)
    get!(fittedmodels, nm) do
        [fit(MixedModel, f, dataset(nm)) for f in fms[nm]]
    end
end