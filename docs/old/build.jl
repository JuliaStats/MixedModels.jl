using Docile, Lexicon, MixedModels

const api_directory = "api"
const modules = Docile.Collector.submodules(MixedModels)

cd(dirname(@__FILE__)) do

    # Run the doctests *before* we start to generate *any* documentation.
    for m in modules
        failures = failed(doctest(m))
        if !isempty(failures.results)
            println("\nDoctests failed, aborting commit.\n")
            display(failures)
            exit(1) # Bail when doctests fail.
        end
    end

    # Generate and save the contents of docstrings as markdown files.
    index  = Index()
    config = Config(md_subheader = :category, category_order = [:module,    :function, :method, :type,
                                                                :typealias, :macro,    :global])
    for mod in modules
        update!(index, save(joinpath(api_directory, "$(mod).md"), mod, config))
    end
    save(joinpath(api_directory, "index.md"), index, config)

end
