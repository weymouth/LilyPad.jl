import WaterLily

@inline _clamp_quad(x::SVector{D,T}, f) where {D,T} =
    SVector{D,T}(clamp(x[d], T(0.5), T(size(f, d) - 1.5)) for d in 1:D)

@inline function _clamp_quad_face(x::SVector{D,T}, f, i::Int) where {D,T}
    y = x + SVector{D,T}(ifelse(i == j, T(0.5), zero(T)) for j in 1:D)
    SVector{D,T}(clamp(y[d], T(0.5), T(size(f, d) - 3) + ifelse(d == i, T(0.5), zero(T))) for d in 1:D)
end

@inline function _quad_weight(t, s::Int)
    s == -1 && return t * (t - 1) / 2
    s == 0 && return 1 - t * t
    return t * (t + 1) / 2
end

@inline function quad(x::SVector{D,T}, f) where {D,T}
    xq = _clamp_quad(x, f)
    xh = xq .+ T(1.5)
    c = ntuple(d -> clamp(round(Int, xh[d]), 2, size(f, d) - 1), D)
    t = ntuple(d -> xh[d] - T(c[d]), D)

    acc = zero(eltype(f))
    for J in CartesianIndices(ntuple(_ -> 3, D))
        s = ntuple(d -> J.I[d] - 2, D)
        I = CartesianIndex(ntuple(d -> c[d] + s[d], D))

        w = one(T)
        @inbounds for d in 1:D
            w *= _quad_weight(t[d], s[d])
        end
        @inbounds acc += f[I] * w
    end
    return acc
end

@inline quad(x::SVector{D,T}, f, i::Int) where {D,T} = quad(_clamp_quad_face(x, f, i), f)