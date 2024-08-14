module IssueData
    using Arrow
    using CSV
    using DataFrames
    using Downloads
    using Scratch
    using ZipFile

    export get_data

    const CACHE = Ref("")
    const URL = "https://github.com/user-attachments/files/16604579/testdataforjulia_bothcase.zip"

    function extract_csv(zipfile, fname; delim=',', header=1, kwargs...)
        file = only(filter(f -> endswith(f.name, fname), zipfile.files))
        return CSV.read(file, DataFrame; delim, header, kwargs...)
    end

    function get_data()
        path = joinpath(CACHE[], "780.arrow")

        isfile(path) && return DataFrame(Arrow.Table(path); copycols=true)

        @info "downloading..."
        data = open(Downloads.download(URL), "r") do io
            zipfile = ZipFile.Reader(io)
            @info "extracting..."
            return extract_csv(zipfile, "testdataforjulia_bothcase.csv")
        end

        transform!(data, :individual_local_identifier => ByRow(string); renamecols=false),
        Arrow.write(path, data)
        return data
    end

    clear_scratchspaces!() = rm.(readdir(CACHE[]))

    function __init__()
        CACHE[] = get_scratch!(Main, "780")
        return nothing
    end
end

using DataFrames
using .IssueData
using LinearAlgebra
using MixedModels
using Statistics

data = get_data()
m0form = @formula(case ~ 0 + Analysisclass + (1|cropyear/individual_local_identifier))

# fails
model = fit(MixedModel, m0form, data, Bernoulli();
            wts=float.(data.weights),
            contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
            fast=false,
            progress=true,
            verbose=false)

# works, non singular, FE look okay
model = fit(MixedModel, m0form, data, Bernoulli();
            wts=float.(data.weights),
            contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
            init_from_lmm=[:θ],
            fast=false,
            progress=true,
            verbose=false)

# works, singular and has questionable FE
m0fast = fit(MixedModel, m0form, data, Bernoulli();
             wts=float.(data.weights),
             contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
             fast=true,
             progress=true,
             verbose=false)

# this model is singular in cropyear, but it looks like there is proper nesting:
groups = select(data, :cropyear, :individual_local_identifier)
unique(groups)
unique(groups, :cropyear)
unique(groups, :individual_local_identifier)

# the estimates for `Nonhabitat_Wet_day` and `Nonhabitat_Wet_night` are identical,
# which seems suspicious, and they have very large standard errors. I think
# this hints at undetected collinearity.
X = modelmatrix(m0fast)
rank(X) # =12
idx = findall(coefnames(m0fast)) do x
    return x in ("Analysisclass: Nonhabitat_Wet_day", "Analysisclass: Nonhabitat_Wet_night")
end

cols = X[:, idx]
# AHA 98% of values are identical because these measurements are very sparse
mean(cols[:, 1] .== cols[:, 2])
mean(cols[:, 1])
mean(cols[:, 2])

counts = sort!(combine(groupby(data, :Analysisclass), nrow => :n), :n)
transform!(counts, :n => ByRow(x -> round(100x / sum(counts.n); digits=1)) => "%")

# let's try reparameterizing

transform!(data, :Analysisclass => ByRow(ac -> NamedTuple{(:habitat, :wet, :time)}(split(ac, "_"))) => AsTable)

m1form =  @formula(case ~ 0 + habitat * wet * time + (1|cropyear & individual_local_identifier))

# fails really fast with a PosDefException
m1fast = fit(MixedModel, m1form, data, Bernoulli();
             wts=float.(data.weights),
             fast=true,
             progress=true,
             verbose=false)

# still fails
m1 = fit(MixedModel, m1form, data, Bernoulli();
         wts=float.(data.weights),
         fast=false,
         progress=true,
         verbose=false)
