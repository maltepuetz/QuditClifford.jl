
"""
    StabilizerTableau(d::Int64, n::Int64, tableau::Array{Int64,2}, storephase::Bool)
    StabilizerTableau(d::Int64, n::Int64, tableau::Array{Int64,2})
    StabilizerTableau(d::Int64, tableau::Array{Int64,2})

A stabilizer tableau for a system of qudits. The generators of the stabilizer state are
stored in the columns of the tableau (because Julia uses column-major order for arrays).
"""
struct StabilizerTableau{T<:InverseMod}
    d::Int64  # qudit dimension
    n::Int64  # number of qudits
    tableau::Array{Int64,2}             # the tableau itself
    storephase::Bool                    # true if the tableau stores the phase information, false otherwise
    workspace::Array{Int64,2}           # workspace for intermediate calculations
    generator_workspace::Vector{Int64}  # workspace for intermediate calculations
    inversemod::T


    function StabilizerTableau(
        d::Int64,
        n::Int64,
        tableau::Array{Int64,2},
        storephase::Bool,
        inversemod::T,
    ) where T<:InverseMod

        if !Primes.isprime(d)
            throw(ArgumentError("Qudit dimension d must be a prime number."))
        end
        if size(tableau, 2) != n || size(tableau, 1) != 2 * n + storephase
            throw(ArgumentError("Tableau dimensions do not match the number of qudits and phase storage."))
        end
        new{T}(d, n, tableau, storephase, zeros(Int64, n, n), zeros(Int64, 2*n), inversemod)
    end
    function StabilizerTableau(
        d::Int64,
        n::Int64,
        tableau::Array{Int64,2},
        storephase::Bool,
    )
        return StabilizerTableau(d, n, tableau, storephase, InverseMod(d))
    end
    function StabilizerTableau(d::Int64, n::Int64, tableau::Array{Int64,2})
        return StabilizerTableau(d, n, tableau, size(tableau, 1) == 2 * n + 1)
    end

    function StabilizerTableau(d::Int64, tableau::Array{Int64,2})
        n = size(tableau, 2)
        return StabilizerTableau(d, n, tableau)
    end
end


# give the struct a nice standart presentation
function Base.show(io::IO, stabtab::StabilizerTableau)
    println(io, "Stabilizer Tableau:")
    println(io, "    Qudit dimension:  d = ", stabtab.d)
    println(io, "    Number of Qudits: n = ", stabtab.n)

    if stabtab.n >= 20 # if the tableau is too large, don't print it
        println(io, "    Tableau is too large to display.")
        return
    end

    println(io, "    Tableau:")


    N = log10(maximum(stabtab.tableau)) |> floor |> Int
    extraspace = 0
    isodd(N) && isodd(stabtab.n) && (extraspace += 1)  # ensure that the tableau is nicely aligned
    print(io, "    ")
    print(io, "-"^(stabtab.n * (N + 2) ÷ 2 - 2 + extraspace), " X ", "-"^(stabtab.n * (N + 2) ÷ 2 - 2 + extraspace))
    print(io, extraspace == 1 ? "  " : "   ")
    print(io, "-"^(stabtab.n * (N + 2) ÷ 2 - 2 + extraspace), " Z ", "-"^(stabtab.n * (N + 2) ÷ 2 - 2 + extraspace))

    println(io)

    for i in 1:size(stabtab.tableau, 2)
        print(io, "   ")
        for j in 1:size(stabtab.tableau, 1)
            ((j == stabtab.n + 1) || (j == 2 * stabtab.n + 1)) && print(io, " |")  # add a separator for the phase column
            print(io, lpad(stabtab.tableau[j, i], N + 2))
        end
        println(io)
    end
end
