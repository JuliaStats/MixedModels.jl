using CairoMakie
using CSV
using DataFrames
using Downloads
using MixedModels
using MixedModelsMakie

const CSV_URL = "https://github.com/JuliaStats/MixedModels.jl/files/9659213/web_areas.csv"

data = CSV.read(Downloads.download(CSV_URL), DataFrame)

contrasts = Dict(:species => Grouping())

form = @formula(web_area ~ 1 + rain + placement + canopy + understory + size + (1|species))

fm1 = fit(MixedModel, form, data; contrasts)

# does look like a bit of heteroskedacity
plot(fitted(fm1), residuals(fm1))

form_log = @formula(log(web_area) ~ 1 + rain + placement + canopy + understory + size + (1|species))

fm1_log = fit(MixedModel, form_log, data; contrasts)

# looks much better
plot(fitted(fm1_log), residuals(fm1_log))

density(residuals(fm1_log))

# looks pretty good
let f = Figure()
    ax = Axis(f[1,1]; aspect=1)
    scatter!(ax, fitted(fm1_log), response(fm1_log))
    ablines!(ax, 0, 1; linestyle=:dash)
    xlims!(ax, -1.4, 3.4)
    ylims!(ax, -1.4, 3.4)
    f
end


# what about sqrt? since we're dealing with areas

form_sqrt = @formula(sqrt(web_area) ~ 1 +  rain + placement + canopy + understory + size + (1  |species))

fm1_sqrt = fit(MixedModel, form_sqrt, data; contrasts)

# not nearly as good as log
plot(fitted(fm1_sqrt), residuals(fm1_sqrt))

density(residuals(fm1_sqrt))

# doesn't look bad
let f = Figure()
    ax = Axis(f[1,1]; aspect=1)
    scatter!(ax, fitted(fm1_sqrt), response(fm1_sqrt))
    ablines!(ax, 0, 1; linestyle=:dash)
    xlims!(ax, 0, 6)
    ylims!(ax, 0, 6)
    f
end

# what about reciprocal/inverse? this often works quite nicely for things where log also works

form_inv = @formula(1 / web_area ~ 1 + rain + placement + canopy + understory + size + (1|species))

fm1_inv = fit(MixedModel, form_inv, data; contrasts)

# othis actually looks kinda bad
plot(fitted(fm1_inv), residuals(fm1_inv))

density(residuals(fm1_inv))

# this almost looks like there are other things we're not controlling for
let f = Figure()
    ax = Axis(f[1,1]; aspect=1)
    scatter!(ax, fitted(fm1_inv), response(fm1_inv))
    ablines!(ax, 0, 1; linestyle=:dash)
    f
end

# one key thing to note here is that there is hole in all the fitted vs. observed plots --
# I suspect there is some type of jump, maybe between species?
