import WaterLily: @loop, inside_u, loc, interp

"""
    departure(x, u, dt)

RK2 characteristic backtrack using velocity field `u`. Returns the departure point.
"""
@inline function departure(x, u, dt) 
    xm = x - (dt / 2) * interp(x, u)
    return x - dt * interp(xm, u)
end

"""
    scalar_advect!(f, φ_src, u, dt, i)

Advect face-component scalar field `φ_src` into `f` for staggered direction
`i` using backtracking from velocity field `u`.
"""
@inline scalar_advect!(f::AbstractArray{T}, φ_src, u, dt, i) where T = @loop f[I] = quad(departure(loc(i, I, T), u, dt), φ_src, i) over I ∈ inside_u(size(f), i)
