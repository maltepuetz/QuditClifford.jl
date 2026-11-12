using QuditClifford
using Documenter
using DocumenterVitepress

include("figure_theme.jl")

DocMeta.setdocmeta!(QuditClifford, :DocTestSetup, :(using QuditClifford); recursive=true)

makedocs(;
    modules=[QuditClifford],
    authors="Malte Pütz",
    sitename="QuditClifford.jl",
    checkdocs=:exports,
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/maltepuetz/QuditClifford.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="https://maltepuetz.github.io/QuditClifford.jl",
        # Default meta description for every page (the default is the generic
        # "Documentation for QuditClifford.jl"); the home page sets its own.
        description="Qudit and qubit Clifford circuit simulation in Julia with stabilizer tableaux for prime dimensions.",
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting-started.md",
            "Examples" => "examples.md",
            "Representation and Conventions" => "conventions.md",
            "Measurements and Expectations" => "measurements.md",
            "Clifford Unitaries" => "unitaries.md",
            "Inspecting States" => "dense-states.md",
        ],
        "API Reference" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/maltepuetz/QuditClifford.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
