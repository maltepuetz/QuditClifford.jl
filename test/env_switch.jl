# Applies the test configuration switches `QC_TEST_LAYOUT` and `QC_TEST_ELTYPE`.
# test/runtests.jl includes this file, and so does every test file that builds
# tableaux, so a test file run on its own honours the switches too. Including
# it again changes nothing.
using QuditClifford

# The layout every constructor uses without a `layout` keyword, so the whole
# suite runs generator-major (the default) or qudit-major.
let layout = get(ENV, "QC_TEST_LAYOUT", "generator")
    layout in ("generator", "qudit") ||
        error("QC_TEST_LAYOUT must be \"generator\" or \"qudit\", got \"$layout\"")
    QuditClifford._DEFAULT_LAYOUT[] = Symbol(layout)
end

# The element type every d = 2 constructor uses without an `eltype` keyword, so
# the whole suite runs on `UInt8` entries (the default) or on packed `Bits`.
let eltype = get(ENV, "QC_TEST_ELTYPE", "uint8")
    eltype in ("uint8", "bits") ||
        error("QC_TEST_ELTYPE must be \"uint8\" or \"bits\", got \"$eltype\"")
    # The shipped method is deleted before the replacement is defined, so no
    # method is overwritten (which `Pkg.test` would report as a warning). Once
    # replaced, the method is left alone.
    if eltype == "bits" && Base.invokelatest(QuditClifford._default_qubit_eltype) !== Bits
        Base.delete_method(which(QuditClifford._default_qubit_eltype, ()))
        @eval QuditClifford _default_qubit_eltype() = Bits
    end
end
