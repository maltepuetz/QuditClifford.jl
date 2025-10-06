struct Operator{T<:AbstractVector{Int64}}
    string::T
end


Base.length(op::Operator) = length(op.string)
function Random.rand!(op::Operator, d::Int)
    isunity = true
    @inbounds for i in eachindex(op.string)
        exponent = rand(0:d-1)
        isunity && (exponent != 0) && (isunity = false)
        op.string[i] = exponent
    end
    isunity && return rand!(op, d)
    return op
end
