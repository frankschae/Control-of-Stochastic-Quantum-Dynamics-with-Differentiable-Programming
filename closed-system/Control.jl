# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# load packages
using Flux, DiffEqFlux, DiffEqSensitivity
using DifferentialEquations
using Plots
using BSON
using Statistics
using ReverseDiff, Zygote, ForwardDiff
using LinearAlgebra
using Random
using BenchmarkTools, Test

#################################################
# read parameters from command line
lr = 0.0015f0 #parse(Float64,ARGS[1]) #0.05
epochs = 400 #parse(Int,ARGS[2]) #100
slurmidx = 1 #parse(Int,ARGS[3]) #1

numtraj = 256 # number of trajectories in parallel simulations for training
numtrajplot = 256 # .. for plotting

# time range for the solver
dt = 0.01f0 #0.001f0
tinterval = 0.02f0
tstart = 0.0f0
Nintervals = 150 # total number of intervals, total time = t_interval*Nintervals
tspan = (tstart,tinterval*Nintervals)

# Hamiltonian parameters
Δ = 20.0f0
Ωmax = 10.0f0 # control parameter (maximum amplitude)

# loss hyperparameters
C1 = Float32(1.0)  # evolution state fidelity
C2 = Float32(0.0) # action amplitudes
C3 = Float32(0.0) # evolution state fidelity for last few steps!

struct Parameters{flType,intType,tType}
	lr::flType
	epochs::intType
	numtraj::intType
	numtrajplot::intType
	dt::flType
	tinterval::flType
	tspan::tType
	Nintervals::intType
	Δ::flType
	Ωmax::flType
	C1::flType
	C2::flType
	C3::flType
end

myparameters = Parameters{typeof(dt),typeof(numtraj), typeof(tspan)}(
  lr, epochs, numtraj, numtrajplot, dt, tinterval, tspan, Nintervals,
  Δ, Ωmax, C1, C2, C3)


################################################
# Define Neural Network

# state-aware
nn = FastChain(
	FastDense(4, 256, relu, initW = Flux.glorot_uniform, initb = Flux.glorot_uniform),
	FastDense(256, 64, relu, initW = Flux.glorot_uniform, initb = Flux.glorot_uniform),
	FastDense(64, 1, softsign, initW = Flux.glorot_uniform, initb = Flux.glorot_uniform))
p_nn = initial_params(nn)

###############################################
# initial state anywhere on the Bloch sphere
function prepare_initial(dt, n_par)
  # shape 4 x n_par
  # input number of parallel realizations and dt for type inference
  # random position on the Bloch sphere
  theta = acos.(2*rand(typeof(dt),n_par).-1)  # uniform sampling for cos(theta) between -1 and 1
  phi = rand(typeof(dt),n_par)*2*pi  # uniform sampling for phi between 0 and 2pi
  # real and imaginary parts ceR, cdR, ceI, cdI
  u0 = [cos.(theta/2), sin.(theta/2).*cos.(phi), false*theta, sin.(theta/2).*sin.(phi)]
  return vcat(transpose.(u0)...) # build matrix
end

# target state
# ψtar = |up>

u0 = prepare_initial(myparameters.dt, myparameters.numtraj)

###############################################
# Define ODE

function qubit(u,p,t)
  # expansion coefficients |Ψ> = ce |e> + cd |d>
  ceR, cdR, ceI, cdI = u # real and imaginary parts

  # Δ: atomic frequency
  # Ω: Rabi frequency for field in x direction
  Δ = p[end-1]
  Ωmax = p[end]
  nn_weights = p[1:end-2]

  Ω = (nn(u, nn_weights).*Ωmax)[1]

  dceR =  1//2*(ceI*Δ+cdI*Ω)
  dcdR = -cdI*Δ/2 +ceI*Ω/2
  dceI = 1//2*(-ceR*Δ-cdR*Ω)
  dcdI = cdR*Δ/2-ceR*Ω/2

  return [dceR, dcdR, dceI, dcdI]
end


# normalization callback

condition(u,t,integrator) = true
function affect!(integrator)
  integrator.u=integrator.u/norm(integrator.u)
end
cb = DiscreteCallback(condition,affect!,save_positions=(false,false))


# get control pulses
p_all = [p_nn; myparameters.Δ; myparameters.Ωmax]
# define ODE problem
prob = ODEProblem{false}(qubit, vec(u0[:,1]), myparameters.tspan, p_all,
   callback=cb
   )

#########################################
# compute loss
function g(u,p,t)
  ceR = @view u[1,:,:]
  cdR = @view u[2,:,:]
  ceI = @view u[3,:,:]
  cdI = @view u[4,:,:]
  p[1]*mean((cdR.^2 + cdI.^2) ./ (ceR.^2 + cdR.^2 + ceI.^2 + cdI.^2))
end

function loss(p, u0, myparameters::Parameters; sensealg = ForwardDiffSensitivity())
  pars = [p; myparameters.Δ; myparameters.Ωmax]

  function prob_func(prob, i, repeat)
    # prepare initial state and applied control pulse
	u0tmp = deepcopy(vec(u0[:,i]))
    remake(prob,
		p = pars,
		u0 = u0tmp,
		callback = cb
		)
  end

  ensembleprob = EnsembleProblem(prob,
   prob_func = prob_func,
   safetycopy = true
   )

  _sol = solve(ensembleprob, Tsit5(), EnsembleThreads(),
    sensealg=sensealg,
    saveat=myparameters.tinterval,
    dt=myparameters.dt,
    adaptive=true, abstol=1e-6, reltol=1e-6,
    trajectories=myparameters.numtraj, batch_size=myparameters.numtraj)

  A = convert(Array,_sol)
  loss = g(A,[myparameters.C1,myparameters.C2,myparameters.C3],nothing)

  return loss
end

#########################################
# visualization -- run for new batch
function visualize(p, u0, myparameters::Parameters; all_traj = true)
  # # initialization
  u = deepcopy(u0)

  pars = [p; myparameters.Δ; myparameters.Ωmax]

  function prob_func(prob, i, repeat)
    # prepare initial state and applied control pulse
	remake(prob,
		p = pars,
		u0 = vec(u0[:,i]),
		callback = cb
		)
  end

  ensembleprob = EnsembleProblem(prob,
   prob_func = prob_func,
   safetycopy = true
   )

  u = solve(ensembleprob, Tsit5(), ensemblealg=EnsembleThreads(),
	saveat=myparameters.tinterval,
	dt=myparameters.dt,
	adaptive=true, abstol=1e-6, reltol=1e-6,
	trajectories=myparameters.numtrajplot, batch_size=myparameters.numtrajplot)


  ceR = @view u[1,:,:]
  cdR = @view u[2,:,:]
  ceI = @view u[3,:,:]
  cdI = @view u[4,:,:]
  infidelity = @. cdR^2 + cdI^2 / (ceR^2 + cdR^2 + ceI^2 + cdI^2)
  meaninfidelity = mean(infidelity)
  loss = myparameters.C1*meaninfidelity

  @info "Loss: " loss

  fidelity = @. ceR^2 + ceI^2 / (ceR^2 + cdR^2 + ceI^2 + cdI^2)

  mf = mean(fidelity, dims=2)[:]
  sf = std(fidelity, dims=2)[:]

  # re-compute actions
  arrayu = Array(u)
  Ωlist = []
  for i = 1:(size(arrayu)[2])
    Ω = vec(nn(arrayu[:,i,:],p).*myparameters.Ωmax)
    push!(Ωlist, Ω)
  end
  Ωlist = hcat(Ωlist...)

  ma = mean(Ωlist, dims=1)[:]
  sa = std(Ωlist, dims=1)[:]

  pl1 = plot(0:myparameters.Nintervals, mf,
		ribbon = sf,
		ylim = (0,1), xlim = (0,myparameters.Nintervals),
		c=1, lw = 1.5, xlabel = "steps", ylabel="Fidelity", legend=false)
  pl2 = plot(0:myparameters.Nintervals, ma,
		ribbon = sa,
		ylim=(-myparameters.Ωmax,myparameters.Ωmax), xlim = (0,myparameters.Nintervals),
		c=2, lw = 1.5, xlabel = "steps", ylabel="Ω(t)", legend=false)
  if all_traj
     plot!(pl1, fidelity, legend=false, c=:gray, alpha=0.1)
     plot!(pl2, Ωlist', legend=false, c=:gray, alpha=0.1)
  else
     plot!(pl1, 0:myparameters.Nintervals, fidelity[:,end],  c=:gray, lw = 1.5, legend=false)
     plot!(pl2, 0:myparameters.Nintervals, Ωlist[end,:], c=:gray, lw = 1.5, legend=false)
  end

  pl = plot(pl1, pl2, layout = (1, 2), legend = false, size=(800,360))
  return pl, loss
end

###################################
# training loop

# optimize the parameters for a few epochs with ADAM on time span Nint
opt = ADAM(myparameters.lr)
list_plots = []
losses = []
for epoch in 1:myparameters.epochs
  println("epoch: $epoch / $(myparameters.epochs)")
  local u0 = prepare_initial(myparameters.dt, myparameters.numtraj)
  _dy, back = @time Zygote.pullback(p -> loss(p, u0, myparameters,
  	sensealg=InterpolatingAdjoint()), p_nn)
  gs = @time back(one(_dy))[1]
  push!(losses, _dy)
  if epoch % myparameters.epochs == 0
    # plot every xth epoch
    local u0 = prepare_initial(myparameters.dt, myparameters.numtrajplot)
    pl, test_loss = visualize(p_nn, u0, myparameters)
    println("Loss (epoch: $epoch): $test_loss")
    display(pl)
    push!(list_plots, pl)
  end
  Flux.Optimise.update!(opt, p_nn, gs)
  println("")
end

# plot training loss
pl = plot(losses, lw = 1.5, xlabel = "some epochs", ylabel="Loss", legend=false)

###################################
# Serialization

bson("Data/ODEControlCont-epochs="*string(myparameters.epochs)*"_numtraj="*join(string.([myparameters.numtraj,myparameters.numtrajplot]), '_')*"_"*string(slurmidx)*".bson",
  Dict(:losses => losses, :popt => p_nn, :lr =>myparameters.lr, :epoch =>myparameters.epochs))

savefig(pl,"./Figures/ODELossCont_epochs="*string(myparameters.epochs)*"_numtraj="*join(string.([myparameters.numtraj,myparameters.numtrajplot]), '_')*"_"*string(slurmidx)*".png")



for (i,plt) in enumerate(list_plots)
  savefig(plt,"./Figures/ODEControlCont"*string(i)*"_epochs="*string(myparameters.epochs)*"_numtraj="*join(string.([myparameters.numtraj,myparameters.numtrajplot]), '_')*"_"*string(slurmidx)*".png")
end

exit()
