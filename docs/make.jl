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
        "Getting Started" => "getting-started.md",
        "Manual" => [
            "Representation and Conventions" => "manual/conventions.md",
            "Measurements and Expectations" => "manual/measurements.md",
            "Clifford Unitaries" => "manual/unitaries.md",
            "Inspecting States" => "manual/dense-states.md",
        ],
        "Examples" => [
            "A two-qutrit measurement" => "examples/qutrit-measurement.md",
            "Entanglement through a measurement" => "examples/ghz-measurement.md",
            "Entropy across every cut" => "examples/entropy-cuts.md",
            "Correlations in mixed states" => "examples/mixed-correlations.md",
            "Purification by measurements" => "examples/purification.md",
            "Boundary dephasing" => "examples/boundary-dephasing.md",
            "Measurement-only Ising circuit" => "examples/measurement-only-ising.md",
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
