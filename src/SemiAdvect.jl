import WaterLily: @loop, inside_u, loc, interp

"""
    departure(x, [u⁰,] u, dt)

Characteristic backtrack for the departure point using RK2 [or Crank-Nicolson]
"""
@inline function departure(x, u, dt) 
    xm = x - dt/2 * interp(x, u)
    return x - dt * interp(xm, u)
end
@inline function departure(x, u⁰, u, dt) 
    dx =  -dt * interp(x, u)
    dx⁰ =  -dt * interp(x+dx, u⁰)
    return x + (dx + dx⁰) / 2
end

"""
    scalar_advect!(f, φ_src, [u⁰,] u, dt, i)

Semi-Lagrangian advection of face-component scalar field `φ_src` at face `i`, overwriting `f`.
"""
@inline scalar_advect!(f::AbstractArray{T}, φ_src, u, dt, i) where T = @loop f[I] = quad(departure(loc(i, I, T), u, dt), φ_src, i) over I ∈ inside_u(size(f), i)
@inline scalar_advect!(f::AbstractArray{T}, φ_src, u⁰, u, dt, i) where T = @loop f[I] = quad(departure(loc(i, I, T), u⁰, u, dt), φ_src, i) over I ∈ inside_u(size(f), i)
