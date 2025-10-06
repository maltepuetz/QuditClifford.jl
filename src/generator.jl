struct Generator{T<:AbstractVector{Int64}}
    string::T
    function Generator(stabtab::StabilizerTableau, i::Int64)
        string = view(stabtab.tableau, :, i)
        new{typeof(string)}(string)
    end
    function Generator(generator_workspace::Vector{Int64})
        new{typeof(generator_workspace)}(generator_workspace)
    end
end

Base.length(gen::Generator) = length(gen.string)

function commutation(gen::Generator, op::Operator)
    # get number of qudits
    N = length(op) ÷ 2

    comm = 0
    for i in 1:N
        comm += gen.string[i] * op.string[i + N]
    end
    for i in 1:N
        comm -= gen.string[i + N] * op.string[i]
    end
    return comm
end
function commutation(gen1::Generator, gen2::Generator)
    # get number of qudits
    N = length(gen1) ÷ 2

    comm = 0
    for i in 1:N
        comm += gen1.string[i] * gen2.string[i + N]
    end
    for i in 1:N
        comm -= gen1.string[i + N] * gen2.string[i]
    end
    return comm
end

function copyto!(gen::Generator, op::Operator)
    for i in eachindex(gen.string)
        gen.string[i] = op.string[i]
    end
end
function copyto!(dest::Generator, src::Generator)
    for i in eachindex(dest.string)
        dest.string[i] = src.string[i]
    end
end

"""
    multiply!(generator::T, op::Operator, power::Int, d::Int) where T <: AbstractVector

Multiplies a generator by another generator to the power of `power` modulo `d`.
The operation is performed in-place, modifying the first generator.
"""
function multiply!(gen1::Generator, gen2::Generator, power::Int, d::Int)
    # @assert power != 0
    for i in eachindex(gen1.string)
        gen1.string[i] = mod(gen1.string[i] + power * gen2.string[i], d)
    end
end
