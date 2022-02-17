using JuliaFormatter

function main()
    perfect = true
    # note: keep in sync with `.github/workflows/format-check.yml`
    for d in ["src/", "test/", "docs/"]
        @info "...linting $d ..."
        dir_perfect = format(d; style=BlueStyle(), join_lines_based_on_source=true)
        perfect = perfect && dir_perfect
    end
    if perfect
        @info "Linting complete - no files altered"
    else
        @info "Linting complete - files altered"
        run(`git status`)
    end
    return nothing
end

main()
