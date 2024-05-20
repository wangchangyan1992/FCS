using Distributed
addprocs(60)

@everywhere begin

include(joinpath(@__DIR__, "tools.jl"))
using ITensors
using JLD2
using ProgressMeter
using Dates

struct GroundState
    t::Float64
    nsites::Int64
    ener::Float64
    gstate::MPS
    maxdim::Int64
end

mutable struct Observer <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64
    min_sweep::Int64
    info::AbstractVector

    Observer(energy_tol=1e-10; min_sweep=5) = new(energy_tol, 1000., min_sweep, [])
end

function ITensors.checkdone!(o::Observer;kwargs...)
    sweep  = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy-o.last_energy)/abs(energy) < o.energy_tol && sweep > o.min_sweep
        println("Stopping DMRG after sweep $sweep")
        o.info ∋ (sweep=sweep)
        return true
    end

    o.last_energy = energy
    return false
end

function gState(params)
    (; t, l, cutoff, ener_tol) = params
    filling     = 1
    site_cutoff = 30
    sites       = siteinds("Boson", l, conserve_qns=true, dim=site_cutoff)

    os = OpSum()

    for s in 1:l-1
        os += -t, "adag", s, "a", s+1
        os += -t, "a", s, "adag", s+1
    end

    for s in 1:l
        os += 1/2, "n", s, "n", s
    end

    ham = MPO(os, sites)

    psi0 = MPS(sites, [string(filling) for i = 1:l])
    nsweeps = 100
    maxdim = [10,20,100,200, 400, 1000]
    cutoff = [cutoff]
    obs = Observer(ener_tol)

    energy, psi = dmrg(ham, psi0; nsweeps, maxdim, cutoff, observer=obs)

    # return energy, psi, maximum(linkdims(psi))
    return psi
end

function Region(nsites, len_reg)
    mid = trunc(nsites / 2) |> Int
    half_reg = trunc(len_reg / 2) |> Int
    x0 = mid - half_reg + 1
    x1 = mid + (len_reg - half_reg)
    return Int(x0), Int(x1)
end

function Fcs(params)
    (; state, λ, x0, x1) = params
    sites = siteinds(state)
    gates = ITensor[]

    for i in x0:x1
        g_i = exp(im * λ * op("n", sites[i])) * exp(-im * λ)
        gates ∋ g_i
    end

    psi = apply(gates, state)

    return -log(inner(state, psi))

end



filename(name) = joinpath(@__DIR__, name)



end


nsites   = 100
ener_tol = 1e-10
ts       = [0.1, 0.6]
cutoff_s = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
x0, x1   = Region(nsites, 35)
λs       = range(0, 2π, 201)
fcs_s    = []

params = [(t=t, l=nsites, cutoff=cutoff, ener_tol) for cutoff in cutoff_s, t in ts]
gstate_s = pmap(gState, params)

max_dims = maxlinkdim.(gstate_s)
@show max_dims

fcs_s = []
@showprogress for psi in gstate_s
    params = [(state=psi, λ=λ, x0=x0, x1=x1) for λ in λs]
    fcs_s ∋ pmap(Fcs, params)
end

jldsave(filename("data/fcs_bh_cutoffs_$(today()).jld2"), fcs_s=fcs_s, nsites = nsites, ts=ts, ener_tol=ener_tol, cutoff_s=cutoff_s, x0=x0, x1=x1, max_dims=max_dims)






ts       = [0.1, 0.6]
nsites_s = [80, 100, 120]
cutoff   = 1e-10
ener_tol = 1e-10
x0, x1   = Region(nsites, 35)
λs       = range(0, 2π, 201)
fcs_s    = []

params = [(t=t, l=nsites, cutoff=cutoff, ener_tol) for nsites in nsites_s, t in ts]
gstate_s = pmap(gState, params)

max_dims = maxlinkdim.(gstate_s)
@show max_dims

@showprogress for psi in gstate_s
    params = [(state=psi, λ=λ, x0=x0, x1=x1) for λ in λs]
    fcs_s ∋ pmap(Fcs, params)
end

jldsave(filename("data/fcs_bh_nsites_$(today()).jld2"), fcs_s=fcs_s, nsites_s = nsites_s, ts=ts, ener_tol=ener_tol, cutoff=cutoff, x0=x0, x1=x1, max_dims=max_dims)
