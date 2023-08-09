using MixedModels
using MixedModels: dataset

@isdefined(gfms) || const global gfms = Dict(
    :cbpp => [@formula((incid/hsz) ~ 1 + period + (1|herd))],
    :contra => [@formula(use ~ 1+age+abs2(age)+urban+livch+(1|urban&dist)),
                @formula(use ~ 1+urban+(1+urban|dist))], # see #563
    :grouseticks => [@formula(ticks ~ 1+year+ch+ (1|index) + (1|brood) + (1|location))],
    :verbagg => [@formula(r2 ~ 1+anger+gender+btype+situ+(1|subj)+(1|item))],
)

@isdefined(fms) || const global fms = Dict(
    :oxide => [@formula(Thickness ~ 1 + (1|Lot/Wafer)),
               @formula(Thickness ~ 1 + Source + (1+Source|Lot) + (1+Source|Lot&Wafer))],
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
        @formula(strength ~ 1 + (1|batch&cask)),
        @formula(strength ~ 1 + (1|batch/cask)),
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
@isdefined(fittedmodels) || const global fittedmodels = Dict{Symbol,Vector{MixedModel}}(
    :dyestuff => [fit(MixedModel, only(fms[:dyestuff]), dataset(:dyestuff); progress=false)]
);

@isdefined(allfms) || const global allfms = merge(fms, gfms)


if !@isdefined(models)
    function models(nm::Symbol)
        get!(fittedmodels, nm) do
            [fit(MixedModel, f, dataset(nm); progress=false) for f in allfms[nm]]
        end
    end
end
