# The storage layout of a tableau's X and Z blocks. Generator-major
# (`GeneratorMajor`) stores generator `j` in column `j` of every block,
# qudit-major (`QuditMajor`) in row `j`. The layout lives in the field `layout`
# and changes only through `qudit_major!` and `generator_major!`, which
# transpose each block inside its existing padded parent.

# The layout marker of a tableau's X and Z blocks. Code that reaches the blocks
# takes it as the argument after the tableau; an entry point reads it once and
# passes it on, so the call that receives it is split on the two layouts.
_layout_marker(tab::AbstractTableau)::Union{GeneratorMajor, QuditMajor} = tab.layout

_transpose_blocks!(tab::StabilizerTableau) =
    (_transpose_block!(tab.stab_x); _transpose_block!(tab.stab_z); nothing)
function _transpose_blocks!(tab::_BitStab)
    buf = tab.word_workspace
    _transpose_bit_block!(tab.stab_x, buf)
    _transpose_bit_block!(tab.stab_z, buf)
    return nothing
end
function _transpose_blocks!(tab::_BitDestab)
    buf = tab.word_workspace
    _transpose_bit_block!(tab.stab_x, buf)
    _transpose_bit_block!(tab.stab_z, buf)
    _transpose_bit_block!(tab.destab_x, buf)
    _transpose_bit_block!(tab.destab_z, buf)
    return nothing
end
function _transpose_blocks!(tab::DestabilizerTableau)
    _transpose_block!(tab.stab_x)
    _transpose_block!(tab.stab_z)
    _transpose_block!(tab.destab_x)
    _transpose_block!(tab.destab_z)
    return nothing
end

"""
    qudit_major!(tab::AbstractTableau) -> tab

Store the generators of `tab` qudit-major, in place, and return `tab`.

Qudit-major storage keeps generator `j` in row `j` of the X and Z blocks, so
the exponents of one qudit across all generators are contiguous in memory.
The switch transposes the stabilizer blocks (and the destabilizer blocks of a
[`DestabilizerTableau`](@ref)) inside their existing arrays and allocates
nothing. It changes nothing else: the state, the generator basis, the phases,
`m` and every matrix the tableau returns stay the same. On a tableau that is
already qudit-major it does nothing.

The layout affects speed only, never a result. No operation leaves the layout
changed; see [`generator_major!`](@ref) for the way back and
[`tableau_layout`](@ref) to query it. The constructors take the layout as the
`layout` keyword.

# Examples
```jldoctest
julia> tab = StabilizerTableau(3, 4; state = :ghz);

julia> S = stabilizer_matrix(tab);

julia> tableau_layout(qudit_major!(tab))
:qudit

julia> stabilizer_matrix(tab) == S
true
```
"""
function qudit_major!(tab::AbstractTableau)
    tab.layout isa QuditMajor && return tab
    _transpose_blocks!(tab)
    tab.layout = QuditMajor()
    return tab
end

"""
    generator_major!(tab::AbstractTableau) -> tab

Store the generators of `tab` generator-major, in place, and return `tab`.

Generator-major storage, the default, keeps generator `j` in column `j` of the
X and Z blocks, so the exponents of one generator are contiguous in memory.
Like [`qudit_major!`](@ref), the switch transposes the blocks inside their
existing arrays, allocates nothing and changes nothing else. On a tableau that
is already generator-major it does nothing.

# Examples
```jldoctest
julia> tab = DestabilizerTableau(2, 3; state = :product, layout = :qudit);

julia> tableau_layout(generator_major!(tab))
:generator

julia> tab == DestabilizerTableau(2, 3; state = :product)
true
```
"""
function generator_major!(tab::AbstractTableau)
    tab.layout isa GeneratorMajor && return tab
    _transpose_blocks!(tab)
    tab.layout = GeneratorMajor()
    return tab
end

"""
    tableau_layout(tab::AbstractTableau) -> Symbol

The storage layout of `tab`: `:generator` (generator-major, the default) or
`:qudit` (qudit-major).

See [`qudit_major!`](@ref) and [`generator_major!`](@ref).

# Examples
```jldoctest
julia> tableau_layout(StabilizerTableau(3, 2))
:generator

julia> tableau_layout(StabilizerTableau(3, 2; layout = :qudit))
:qudit
```
"""
tableau_layout(tab::AbstractTableau) = tab.layout isa QuditMajor ? :qudit : :generator

# The layout as `show` prints it.
_layout_name(tab::AbstractTableau) = tab.layout isa QuditMajor ? "qudit-major" : "generator-major"

# Switch a freshly built generator-major tableau to the `layout` symbol a
# constructor received (already checked by `_check_layout`).
_apply_layout!(tab::AbstractTableau, layout::Symbol) =
    layout === :qudit ? qudit_major!(tab) : tab
