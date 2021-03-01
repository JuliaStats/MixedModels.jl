# Alternative Output Formats

In the documentation, we have presented the output from MixedModels.jl in the same format you will see when working in the REPL.
You may have noticed, however, that output from other packages received pretty printing.
For example, DataFrames are converted into nice HTML tables.
In MixedModels, we recently (v3.2.0) introduced limited support for such pretty printing.
(For more details on how the print and display system in Julia works, check out [this NextJournal post](https://nextjournal.com/sdanisch/julias-display-system).)

In particular, we have defined Markdown output, i.e. `show` methods, for our types, which can be easily translated into HTML, LaTeX or even a MS Word Document using tools such as [pandoc](https://pandoc.org/).
Packages like `IJulia` and `Documenter` can often detect the presence of these display options and use them automatically.


```@example Main
using MixedModels
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1+days|subj)), MixedModels.dataset(:sleepstudy))
```

Note that the display here is more succinct than the standard REPL display:

```@example Main
using DisplayAs
m1 |> DisplayAs.Text
```

This brevity is intentional: we wanted these types work well with traditional academic publishing constraints on tables.
The summary for a model fit presented in the REPL does not mesh well with being treated a single table (with columns shared between the random and fixed effects).
In our experience, this leads to difficulties in typesetting the resulting tables.
We nonetheless encourage users to report fit statistics such as the log likelihood or AIC as part of the caption of their table.
If the correlation parameters in the random effects are of interest, then [`VarCorr`](@ref) can also be pretty printed:

```@example Main
VarCorr(m1)
```

Similarly for [`BlockDescription`](@ref) and `MixedModels.likelihoodratiotest`:

```@example Main
BlockDescription(m1)
```

```@example Main
m0 = fit(MixedModel, @formula(reaction ~ 1 + (1|subj)), MixedModels.dataset(:sleepstudy))
MixedModels.likelihoodratiotest(m0,m1)
```

To explicitly invoke this behavior, we must specify the right `show` method:
```julia
show(MIME("text/markdown"), m1)
```
```@eval Main
sprint(show, MIME("text/markdown"), m1)
```

(In the future, we may directly support HTML and LaTeX as MIME types.)

This output can also be written directly to file:

```julia
show(open("model.md", "w"), MIME("text/markdown"), m1)
```
