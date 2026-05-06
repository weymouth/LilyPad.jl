import WaterLily: BC!, exitBC!, @loop, inside_u, inside, ∂, μddn

"""
    LilyFlow{D,T,...} <: WaterLily.AbstractFlow{D,T}

Thin wrapper around `WaterLily.Flow` that overrides `mom_predict!`, `mom_correct!` 
`BDIM!` and `CFL` with semi-Lagrangian advection functions.
"""
struct LilyFlow{D, T, Sf<:AbstractArray{T}, Vf<:AbstractArray{T}, Tf<:AbstractArray{T}} <: AbstractFlow{D,T}
    flow :: Flow{D,T,Sf,Vf,Tf}
end
LilyFlow(args...; kwargs...) = LilyFlow(Flow(args...; kwargs...))

Base.getproperty(a::LilyFlow, name::Symbol) =
    name === :flow ? getfield(a, :flow) : getproperty(getfield(a, :flow), name)
Base.setproperty!(a::LilyFlow, name::Symbol, x) =
    setproperty!(getfield(a, :flow), name, x)

"""    mom_predict!(a::LilyFlow, t₀, t₁)

Semi-Lagrangian predictor: self-advect `u⁰`, apply BDIM blending, and enforce BCs at `t₁`.
"""
function WaterLily.mom_predict!(a::LilyFlow{D}, t₀, t₁; kwargs...) where D
    a.f .= a.u⁰; dt = t₁ - t₀
    for i in 1:D
        scalar_advect!(selectdim(a.f, D+1, i), selectdim(a.u⁰, D+1, i), a.u⁰, dt, i)
    end
    BDIM!(a); BC!(a.u, a.uBC, a.exitBC, a.perdir, t₁)
    a.exitBC && exitBC!(a.u, a.u⁰, a.Δt[end])
end

"""    mom_correct!(a::LilyFlow, t)

Semi-Lagrangian corrector: advect `u⁰ - 0.5*dt*μ₀*∇p` under the averaged
velocity `0.5*(u⁰+u_pred)`, apply BDIM blending, and enforce BCs at `t`.
"""
function WaterLily.mom_correct!(a::LilyFlow{D}, t; kwargs...) where D
    a.f .= a.u⁰; dt = a.Δt[end]
    a.u .= (a.u .+ a.u⁰) ./ 2
    for i in 1:D
        @loop a.σ[I] = a.u⁰[I,i] over I ∈ CartesianIndices(a.σ)
        @loop a.σ[I] -= (dt / 2) * a.μ₀[I, i] * ∂(i, I, a.p) over I ∈ inside(a.σ)
        scalar_advect!(selectdim(a.f, D+1, i), a.σ, a.u, dt, i)
    end
    BDIM!(a); BC!(a.u, a.uBC, a.exitBC, a.perdir, t)
end

"""
    CFL(a::LilyFlow)

Semi-Lagrangian step size is unlimited, so return the current timestep `a.Δt[end]`.
"""
WaterLily.CFL(a::LilyFlow; kwargs...) = a.Δt[end]

"""
    BDIM!(a::LilyFlow)

Semi-Lagrangian BDIM blending on the advected velocity field.
"""
function BDIM!(a::LilyFlow)
    @loop a.f[Ii] -= a.V[Ii] over Ii ∈ CartesianIndices(a.f)
    @loop a.u[Ii] = μddn(Ii, a.μ₁, a.f) + a.V[Ii] + a.μ₀[Ii] * a.f[Ii] over Ii ∈ inside_u(size(a.p))
end

