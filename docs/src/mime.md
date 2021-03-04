# Alternative display and output formats

In the documentation, we have presented the output from MixedModels.jl in the same format you will see when working in the REPL.
You may have noticed, however, that output from other packages received pretty printing.
For example, DataFrames are converted into nice HTML tables.
In MixedModels, we recently (v3.2.0) introduced limited support for such pretty printing.
(For more details on how the print and display system in Julia works, check out [this NextJournal post](https://nextjournal.com/sdanisch/julias-display-system).)

In particular, we have defined Markdown, HTML and LaTeX output, i.e. `show` methods, for our types.
Note that the Markdown output can also be easily and more flexibly translated into HTML, LaTeX (e.g. with `booktabs`) or even a MS Word Document using tools such as [pandoc](https://pandoc.org/).
Packages like `IJulia` and `Documenter` can often detect the presence of these display options and use them automatically.


```@example Main
using MixedModels
form = @formula(rt_trunc ~ 1 + spkr * prec * load +
                          (1 + load | item) +
                          (1 + spkr + prec + load | subj))
contr = Dict(:spkr => EffectsCoding(),
             :prec => EffectsCoding(),
             :load => EffectsCoding(),
             :item => Grouping(),
             :subj => Grouping())
kbm = fit(MixedModel, form, MixedModels.dataset(:kb07); contrasts=contr)
```

Note that the display here is more succinct than the standard REPL display:

```@example Main
using DisplayAs
kbm |> DisplayAs.Text
```

This brevity is intentional: we wanted these types to work well with traditional academic publishing constraints on tables.
The summary for a model fit presented in the REPL does not mesh well with being treated as a single table (with columns shared between the random and fixed effects).
In our experience, this leads to difficulties in typesetting the resulting tables.
We nonetheless encourage users to report fit statistics such as the log likelihood or AIC as part of the caption of their table.
If the correlation parameters in the random effects are of interest, then [`VarCorr`](@ref) can also be pretty printed:

```@example Main
VarCorr(kbm)
```

Similarly for [`BlockDescription`](@ref), `OptSummary` and `MixedModels.likelihoodratiotest`:

```@example Main
BlockDescription(kbm)
```

```@example Main
kbm.optsum
```

```@example Main
m0 = fit(MixedModel, @formula(reaction ~ 1 + (1|subj)), MixedModels.dataset(:sleepstudy))
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1+days|subj)), MixedModels.dataset(:sleepstudy))
MixedModels.likelihoodratiotest(m0,m1)
```

To explicitly invoke this behavior, we must specify the right `show` method.
(The raw and not rendered output is intentionally shown here.)
```julia
show(MIME("text/markdown"), m1)
```
```@example Main
println(sprint(show, MIME("text/markdown"), kbm)) # hide
```
```julia
show(MIME("text/html"), m1)
```
```@example Main
println(sprint(show, MIME("text/html"), kbm)) # hide
```
Note for that LaTeX, the column labels for the random effects are slightly changed: σ is placed into math mode and escaped and the grouping variable is turned into a subscript.
Similarly for the likelihood ratio test, the χ² is escaped into math mode.
This transformation improves pdfLaTeX and journal compatibility, but also means that XeLaTeX and LuaTeX may use a different font at this point.
```julia
show(MIME("text/latex"), m1)
```
```@example Main
println(sprint(show, MIME("text/latex"), kbm)) # hide
```
This escaping behavior can be disabled by specifying `"text/xelatex"` as the MIME type.
(Note that other symbols may still be escaped, as the internal conversion uses the `Markdown` module from the standard library, which performs some escaping on its own.)
```julia
show(MIME("text/xelatex"), m1)
```
```@example Main
println(sprint(show, MIME("text/xelatex"), kbm)) # hide
```

This output can also be written directly to file:

```julia
open("model.md", "w") do io
    show(io, MIME("text/markdown"), kbm)
end
```
