##### stored Clifford operators #####
#
# Semantic data are `(d, targets, F, a)`. `image_xdotz` is a derived cache;
# `inv2` and `fast` are arithmetic context resolved once; `v`, `vout`, `zpref`
# are owned scratch. Direct field mutation is unsupported.
#
# Everything in this file dispatches on `CliffordOperator`, which is why the
# file is included AFTER `apply_clifford.jl`: a method signature naming a type
# needs that type to exist, while references inside method bodies resolve later.

"""
    CliffordOperator(d, targets, F, a; check=true)
    CliffordOperator(g::AbstractClifford, d::Int)

A Clifford unitary on an ordered support, stored at a fixed dimension `d`.

The operator is fixed up to a global phase by the conjugation images of the
ordered generators `X_t1 … X_tk, Z_t1 … Z_tk`: column `i` of the `2k × 2k`
symplectic matrix `F` is the exponent vector of the image of generator `i`, in
all-X-then-all-Z order, and `a[i]` is that image's raw phase exponent modulo
`phase_modulus(d)`.

Unlike the named gates, which are dimension-agnostic, a `CliffordOperator`
carries its own `d` and has no register size; targets are checked against the
register only when it meets a tableau or a Pauli.

# Arguments
- `d::Int`: prime qudit dimension.
- `targets`: ordered, distinct, positive qudit indices. Order is preserved.
- `F`: `2k × 2k` integer matrix, normalized modulo `d` on construction.
- `a`: length-`2k` raw phases, normalized modulo `phase_modulus(d)`.

# Keyword Arguments
- `check::Bool=true`: verify `Fᵀ Ω F = Ω (mod d)` and, at `d = 2`, the parity
  `a[i] ≡ x(F[:,i])·z(F[:,i]) (mod 2)`. Passing `false` skips **only** those two
  algebraic checks — shapes, normalization, the cache and independent ownership
  still happen — and makes their preconditions the caller's responsibility.
  Symplecticity is O(k³); the parity pass is O(k) once the mandatory O(k²) cache
  is built. At `d = 2` the parity condition is exactly "every generator image is
  Hermitian", which is what maps physical observables to physical observables.
  Nothing downstream re-checks it: `measure!`'s `phase_policy` inspects only the
  Pauli handed to it, never a stored generator.

# Throws
`ArgumentError` for a non-prime `d`, offset-indexed inputs, malformed shapes,
nonpositive or repeated targets, sizes that do not fit in `Int`, a
non-symplectic `F`, or a qubit parity violation.

# Ownership and concurrency
Construction copies caller-owned arrays. Treat the operator's fields as
read-only: mutating `targets`, `F`, `a`, or derived data directly is unsupported
and can invalidate cached context and the assumptions of `apply!`. Construct
a new operator to change its action. `copy(U)` owns independent arrays.

`apply!` reuses the operator's scratch, so concurrent applications must use
independent copies. Allocating `conjugate`, `inv`, and composition leave operand
semantic data and scratch untouched.

# Examples
```julia
F = [0 1; 1 0]                     # Fourier at d = 2, as a stored operator
U = CliffordOperator(2, [1], F, [0, 0])
apply!(tab, U)
inv(U)
```

See also [`AbstractClifford`](@ref), [`apply!`](@ref), [`conjugate`](@ref).
"""
struct CliffordOperator <: AbstractClifford
    d::Int
    targets::Vector{Int}
    F::Matrix{Int}
    a::Vector{Int}
    image_xdotz::Vector{Int}
    inv2::Int
    fast::Bool
    v::Vector{Int}
    vout::Vector{Int}
    zpref::Vector{Int}

    # Internal owned-data boundary. Accepts ONLY package-produced arrays that
    # are already independent, canonical, correctly shaped and algebraically
    # trusted under the caller's own contract -- which includes copying an
    # operator originally built with `check=false`, so that copying never
    # silently reinstates the skipped O(k^3) check. Defining any inner
    # constructor suppresses Julia's default full-field one, which is the point.
    global function _owned_clifford_operator(d::Int, targets::Vector{Int},
                                             F::Matrix{Int}, a::Vector{Int},
                                             D::Vector{Int}, inv2::Int,
                                             fast::Bool, v::Vector{Int},
                                             vout::Vector{Int},
                                             zpref::Vector{Int})
        return new(d, targets, F, a, D, inv2, fast, v, vout, zpref)
    end
end

# Ω-pairing of two columns of a 2k x 2k matrix, with overflow-safe arithmetic.
@inline function _symplectic_pair(F::Matrix{Int}, i::Int, j::Int, k::Int,
                                  d::Int, fast::Bool)
    if fast
        s = 0
        @inbounds for q in 1:k
            s += F[q, i] * F[k + q, j] - F[k + q, i] * F[q, j]
        end
        return mod(s, d)
    end
    s = 0
    @inbounds for q in 1:k
        s = add_mod(s, mul_mod(F[q, i], F[k + q, j], d), d)
        s = sub_mod(s, mul_mod(F[k + q, i], F[q, j], d), d)
    end
    return s
end

function _check_symplectic(F::Matrix{Int}, k::Int, d::Int, fast::Bool)
    S = 2k
    for i in 1:S, j in 1:S
        expected = (i <= k && j == i + k) ? 1 :
                   (j <= k && i == j + k) ? mod(-1, d) : 0
        _symplectic_pair(F, i, j, k, d, fast) == expected && continue
        throw(ArgumentError(
            "F is not symplectic mod $d: the Ω-pairing of columns $i and $j is " *
            "$(_symplectic_pair(F, i, j, k, d, fast)), expected $expected. " *
            "Pass check=false only if you can guarantee symplecticity yourself."))
    end
    return nothing
end

function _image_xdotz_dense(F::Matrix{Int}, k::Int, d::Int, fast::Bool)
    S = 2k
    D = Vector{Int}(undef, S)
    if fast
        @inbounds for i in 1:S
            s = 0
            for q in 1:k
                s += F[q, i] * F[k + q, i]
            end
            D[i] = mod(s, d)
        end
    else
        @inbounds for i in 1:S
            s = 0
            for q in 1:k
                s = add_mod(s, mul_mod(F[q, i], F[k + q, i], d), d)
            end
            D[i] = s
        end
    end
    return D
end

# Validate each requested array's element and byte count before reading inputs.
# Empty support takes this same path without division by S.
function _clifford_size(k::Int)
    try
        S = Base.checked_mul(2, k)
        Ssq = Base.checked_mul(S, S)
        Base.checked_mul(Ssq, sizeof(Int)) # F
        Base.checked_mul(S, sizeof(Int))   # a, D, v and vout individually
        Base.checked_mul(k, sizeof(Int))   # targets and zpref individually
        return S
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("Clifford support k = $k has an unrepresentable element or byte count."))
    end
end

# Check the i-th target for positivity and against the targets already stored in
# t[1:i-1], then store it. The O(k^2) distinctness scan runs at construction
# only; consumption re-checks bounds alone.
@inline function _store_target!(t::Vector{Int}, i::Int, ti::Int)
    ti > 0 || throw(ArgumentError("Clifford target indices must be positive, got $ti."))
    @inbounds for r in 1:(i - 1)
        t[r] == ti && throw(ArgumentError(
            "Clifford targets must be distinct; $ti appears more than once."))
    end
    t[i] = ti
    return nothing
end

function CliffordOperator(d::Int, targets::AbstractVector{<:Integer},
                          F::AbstractMatrix{<:Integer},
                          a::AbstractVector{<:Integer}; check::Bool = true)
    # 1. dimension
    Primes.isprime(d) || throw(ArgumentError("Qudit dimension d must be a prime number."))
    # 2. one-based axes -- a structural condition even when check=false. A
    #    correctly sized zero-based vector passes every length test and then
    #    invalidates the `1:k` loops below.
    Base.require_one_based_indexing(targets, F, a)
    # 3. sizes, before forming anything. Checked arithmetic so an impossible
    #    shape is an ArgumentError rather than a silent wrap, and so a huge
    #    lazy target range is never traversed.
    k = length(targets)
    S = _clifford_size(k)
    # 4. shapes
    size(F) == (S, S) || throw(ArgumentError(
        "F must be $(S)×$(S) for k = $k targets, got $(size(F))."))
    length(a) == S || throw(ArgumentError(
        "a must have length $S for k = $k targets, got $(length(a))."))
    # 5. target values, then copy into explicitly allocated dense storage.
    #    `copy(targets)` need not return a Vector{Int} -- a range is one example.
    t = Vector{Int}(undef, k)
    @inbounds for i in 1:k
        ti = targets[i]
        (ti isa Integer && typemin(Int) <= ti <= typemax(Int)) ||
            throw(ArgumentError("Clifford target $ti is not representable as Int."))
        _store_target!(t, i, Int(ti))
    end
    p = phase_modulus(d)
    Fc = Matrix{Int}(undef, S, S)
    @inbounds for col in 1:S, row in 1:S
        Fc[row, col] = Int(mod(F[row, col], d))
    end
    ac = Vector{Int}(undef, S)
    @inbounds for i in 1:S
        ac[i] = Int(mod(a[i], p))
    end
    # 6. derived cache and arithmetic context
    fast = clifford_fast_dots(S, d)
    D = _image_xdotz_dense(Fc, k, d, fast)
    # 7. algebraic checks
    if check
        _check_symplectic(Fc, k, d, fast)
        if d == 2
            @inbounds for i in 1:S
                mod(ac[i] - D[i], 2) == 0 || throw(ArgumentError(
                    "Qubit generator image $i is not Hermitian: raw phase $(ac[i]) " *
                    "must match x·z = $(D[i]) modulo 2."))
            end
        end
    end
    # 8. inverse of 2 -- NEVER invmod(2, 2), which is undefined.
    inv2 = d == 2 ? 0 : Base.invmod(2, d)
    return _owned_clifford_operator(d, t, Fc, ac, D, inv2, fast,
                                    zeros(Int, S), zeros(Int, S), zeros(Int, k))
end

# Materialize a named gate. `_clifford_data` performs the modulus-dependent
# gate validation (e.g. a zero-residue Multiplier), so it runs only after the
# structural checks, and with `JustInTimeInvMod` so a standalone operator never
# builds a dimension-sized lookup table.
function CliffordOperator(g::AbstractClifford, d::Int)
    Primes.isprime(d) || throw(ArgumentError("Qudit dimension d must be a prime number."))
    raw = _clifford_targets(g)
    k = length(raw)
    t = Vector{Int}(undef, k)
    @inbounds for i in 1:k
        _store_target!(t, i, Int(raw[i]))
    end
    _, tF, ta = _clifford_data(g, d, JustInTimeInvMod())
    S = 2k
    F = Matrix{Int}(undef, S, S)
    # The tuple form is COLUMNS: tF[col][row]. Writing this the other way round
    # transposes every gate silently.
    @inbounds for col in 1:S, row in 1:S
        F[row, col] = tF[col][row]
    end
    a = Vector{Int}(undef, S)
    @inbounds for i in 1:S
        a[i] = ta[i]
    end
    fast = clifford_fast_dots(S, d)
    D = _image_xdotz_dense(F, k, d, fast)
    inv2 = d == 2 ? 0 : Base.invmod(2, d)
    return _owned_clifford_operator(d, t, F, a, D, inv2, fast,
                                    zeros(Int, S), zeros(Int, S), zeros(Int, k))
end

# Independent copy of a stored operator. Trusted by construction, so this never
# repeats the O(k^3) algebraic check -- copying an operator built with
# `check=false` must not surprise the caller by rejecting it.
function CliffordOperator(U::CliffordOperator, d::Int)
    U.d == d || throw(ArgumentError(
        "CliffordOperator was built for d=$(U.d), but d=$d was requested."))
    k = length(U.targets); S = 2k
    return _owned_clifford_operator(d, copy(U.targets), copy(U.F), copy(U.a),
                                    copy(U.image_xdotz), U.inv2, U.fast,
                                    zeros(Int, S), zeros(Int, S), zeros(Int, k))
end

Base.copy(U::CliffordOperator) = CliffordOperator(U, U.d)

# Semantic data only. Cache and scratch never affect equality or hashing, so an
# operator must not be mutated while it is a dictionary key.
Base.:(==)(U::CliffordOperator, V::CliffordOperator) =
    U.d == V.d && U.targets == V.targets && U.F == V.F && U.a == V.a

Base.isequal(U::CliffordOperator, V::CliffordOperator) =
    isequal(U.d, V.d) && isequal(U.targets, V.targets) &&
    isequal(U.F, V.F) && isequal(U.a, V.a)

Base.hash(U::CliffordOperator, h::UInt) =
    hash(U.a, hash(U.F, hash(U.targets, hash(U.d, hash(:CliffordOperator, h)))))

Base.show(io::IO, U::CliffordOperator) =
    print(io, "CliffordOperator(d=", U.d, ", targets=", U.targets,
          ", k=", length(U.targets), ")")

@inline _clifford_targets(U::CliffordOperator) = U.targets

# Bounds only. Distinctness is proven by the constructor, so re-deriving it
# per application would add an O(k^2) prelude that dominates when `m` is small.
@inline function _validate_clifford_targets(U::CliffordOperator, n::Int)
    t = U.targets
    @inbounds for i in eachindex(t)
        (1 <= t[i] <= n) || throw(ArgumentError(
            "Clifford target $(t[i]) is outside the register 1:$n."))
    end
    return nothing
end

# The ONLY place a prepared view is built over a stored operator. Taking the
# buffers explicitly is what keeps an allocating API from borrowing operand
# scratch: `_prepare` passes `U`'s own buffers, `_prepare_for_conjugation`,
# `inv` and `∘` pass call-local or result-owned ones.
@inline function _dense_view(U::CliffordOperator, storephase::Bool,
                             v::Vector{Int}, vout::Vector{Int},
                             zpref::Vector{Int}; force_safe::Bool = false)
    return PreparedDenseClifford(U.targets, U.F, U.a, U.image_xdotz, v, vout,
                                 zpref, length(U.targets), U.d,
                                 phase_modulus(U.d), U.inv2,
                                 U.fast && !force_safe, storephase)
end

# EVERY positional argument type matches P1's
# `_prepare(::AbstractClifford, ::Int, ::InverseMod, ::Bool)`. Annotating only
# the first argument would leave two methods neither of which is more specific,
# so an ordinary stored `apply!` would raise an ambiguity `MethodError`. Keep
# trailing types aligned on every hook that specializes an `AbstractClifford`
# fallback. The tableau's `InverseMod` is deliberately ignored: the operator
# carries its own `inv2`, and its derived data is never rebuilt per application.
function _prepare(U::CliffordOperator, d::Int, ::InverseMod, storephase::Bool;
                  force_safe::Bool = false)
    U.d == d || throw(ArgumentError(
        "CliffordOperator was built for d=$(U.d), but the tableau has d=$d."))
    return _dense_view(U, storephase, U.v, U.vout, U.zpref;
                       force_safe = force_safe)
end
