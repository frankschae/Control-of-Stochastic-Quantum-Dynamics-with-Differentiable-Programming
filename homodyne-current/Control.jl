# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()


using Flux, DiffEqFlux, DiffEqSensitivity
using DifferentialEquations
using DiffEqSensitivity
using DiffEqNoiseProcess
using Plots
using Printf, BSON
using QuantumOptics
using Statistics
using Zygote
using StaticArrays
using LinearAlgebra

#################################################
#read parameters from external file
include("parameters_qb.jl")
using Main.parameters

struct Parameters
	dim::Int64
	w::Float32
	force_mag::Float32
	n_steps::Int64
	dt::Float32
	n_substeps::Int64
	gamma::Float32
end

MyParameters=Parameters(
	parameters.parameters_para["N"],
	parameters.parameters_para["w"],
	parameters.parameters_para["force_mag"],
	parameters.parameters_para["max_episode_steps"],
	parameters.parameters_para["dt"],
	parameters.parameters_para["n_substeps"],
	parameters.parameters_para["gamma"])

n_par = 64
# loss hyperparameters
C1 = 1.2f0 # evolution state fid
C2 = 0.0005f0 # action amplitudes
C3 = 0.8f0*MyParameters.n_steps/50 #enhance last 50steps

########################
using Random
Random.seed!(2)

################################
# Hamiltonian and evolution
basis = SpinBasis(1//2) #max n=dim-1
σ_z = sigmaz(basis)
σ_x = sigmax(basis)
σ_p = sigmap(basis)
σ_m = sigmam(basis)
σ_pm = σ_p*σ_m

H_0pre = MyParameters.w/2.0f0*σ_z #drift Hamiltonian
H_1pre = σ_x #control field

#to the dense regime, all real, otherwise Flux.train! complains
H_0 = real(Array{Float32}(H_0pre.data))
H_1 = real(Array{Float32}(H_1pre.data))

Arrayσ_p = real(Array{Float32}(σ_p.data))
Arrayσ_m = real(Array{Float32}(σ_m.data))
Arrayσ_pm = real(Array{Float32}(σ_pm.data))

###################
# DRIFT TERM FOR SDE
function qb_dynamics_dt!(du,u, α, t)  #du/u have a standard dimension
	ψRe = u[1:MyParameters.dim]   #(dim,)
	ψIm = u[MyParameters.dim+1:end]

	HRe = H_0 .+ α[1]*H_1 #size (dim,dim) ,alpha is just a single number

	Reρ = ψRe.*transpose(ψRe) + ψIm.*transpose(ψIm)
	ex_x = real(sum(diag((Arrayσ_p+Arrayσ_m)*Reρ)))

	HIm = -Arrayσ_pm/2 + ex_x*Arrayσ_m
	du[1:MyParameters.dim] = dψRe = HIm*ψRe + HRe*ψIm; #size dim)
	du[MyParameters.dim+1:end] = dψIm = HIm*ψIm - HRe*ψRe;
end

###################
# DIFFUSION TERM FOR SDE
function qb_dynamics_dW!(du,u, α, t)  #last action IS NOT STORED IN u
	ψRe = u[1:MyParameters.dim]   #(dim)
	ψIm = u[MyParameters.dim+1:end]

	HRe = Arrayσ_m

	du[1:MyParameters.dim] = dψRe = HRe*ψRe; #size (dim)
	du[MyParameters.dim+1:end] = dψIm = HRe*ψIm;
end


################################################
#model
@info("Constructing model...")
#state-aware, input = homodyne current values as integrated over the last n_substeps
state_1 = Dense(MyParameters.n_substeps, 256,relu ,initb = Flux.glorot_uniform)
state_2 = Dense(256, 256,relu, initb = Flux.glorot_uniform)
state_3 = Dense(256, 128,relu, initb = Flux.glorot_uniform)
#action-aware, input = last actions provided by the NN
nαin = 8  #number of last actions to be remembered
action_1 = Dense(nαin, 128,relu,initb = Flux.glorot_uniform)
action_2 = Dense(128, 128,relu,initb = Flux.glorot_uniform)
#combined
combine_1 = Dense(256, 64,relu,initb = Flux.glorot_uniform)
combine_2 = Dense(64, 32,relu,initb = Flux.glorot_uniform)
combine_3 = Dense(32, 1,softsign,initb = Flux.glorot_uniform)

state_aware=Chain(state_1, state_2, state_3)
action_aware=Chain(action_1,action_2)
combined=Chain(combine_1, combine_2, combine_3)

#this concatenates the full NN
struct Concat{T}
  catted::T
end
Concat(xs...) = Concat(xs)
Flux.@functor Concat

function (C::Concat)(x)
    mapreduce((f, x) -> f(x), vcat, C.catted, x)
end

model = Chain(Concat(state_aware, action_aware), combined);
p1, re = Flux.destructure(model)

###############################################
# initial state anywhere on the Bloch sphere
u0 = Array{Float32,2}(undef, 2*MyParameters.dim,n_par)
fill!(u0,0.0f0)
u0[2,:] .= 1.0f0 #down state

function prepare_initial!(u0) #random position on the Bloch sphere
	fill!(u0,0.0f0)
	theta = acos.(2*rand(n_par).-[1])  #uniform sampling for cos(theta) between -1 and 1
	phi = rand(n_par)*2*pi
	#real parts
	u0[1,:] += cos.(theta/2)
	u0[2,:] += sin.(theta/2).*cos.(phi)
	#imag parts
	#u0[3,:].+=0
	u0[4,:] += sin.(theta/2).*sin.(phi)
	#normalize initial state -already normalized by definition
	norm_factor = sqrt.(sum(u0[1:2*MyParameters.dim,:].^2,dims=1))
	u0[1:2*MyParameters.dim,:] = u0[1:2*MyParameters.dim,:]./norm_factor
	return u0
end

# target state
# ψtar = |up>
ut_complex = Array{Float32,1}(undef, MyParameters.dim)
fill!(ut_complex,0.0f0)
ut_complex[1,:] .= 1.0f0
Re_ut = real(Array{Float32}(ut_complex))
Im_ut = imag(Array{Float32}(ut_complex))
#To visualize
Fock_target = SVector{MyParameters.dim}(Re_ut.^2+Im_ut.^2)

############################################
#Mutate vector outside gradients!
# Necessary for collectiong results during the training
mut(A,index,x) = (A[index] = x)
mut_row(A,index,x) = (A[index,:] = x)
mut_vec(A,x) = (A .=x )
#tell Zygote not to take gradients of these
Zygote.@nograd mut
Zygote.@nograd mut_row
Zygote.@nograd mut_vec

##################################
# time range for the solver
t_interval = round(MyParameters.n_substeps*MyParameters.dt,digits=5)
tspan = (0.0f0,t_interval)

#########################################
#Static Arrays to collect results
mean_fid_store = zeros(MVector{MyParameters.n_steps+1,Float32})
std_fid_store = zeros(MVector{MyParameters.n_steps+1,Float32})
single_traj_fid_store = zeros(MMatrix{MyParameters.n_steps+1,n_par,Float32})
mean_action_store = zeros(MVector{MyParameters.n_steps,Float32})
std_action_store = zeros(MVector{MyParameters.n_steps,Float32})
single_traj_action_store = zeros(MMatrix{MyParameters.n_steps,n_par,Float32})
Fock_end_example = zeros(MVector{MyParameters.dim,Float32})

#########################################
# compute loss
###########################

#normalize the state using callback function
condition(u,t,integrator) = true
function affect!(integrator)
	integrator.u=integrator.u/norm(integrator.u)
end
cb = DiscreteCallback(condition,affect!,save_positions=(false,true))

CreateGrid(W1) =  (NoiseGrid(Array{Float32}((0.0:MyParameters.dt:(t_interval+MyParameters.dt))),W1))
Zygote.@nograd CreateGrid #avoid taking grads of this function

# set scalar random process
W = sqrt(MyParameters.dt)*randn(Float32,MyParameters.n_substeps+1) #for 1 trajectory
W1 = cumsum([zero(MyParameters.dt); W[1:end-1]], dims=1)
NG = CreateGrid(W1)
# define SDE problem
prepare_initial!(u0)
α = zeros(Float32,1,n_par)
prob = SDEProblem{true}(qb_dynamics_dt!,qb_dynamics_dW!, u0[:,1], tspan,α[1,1], noise=NG )

function compute_ex_x(sol)
	#size sol (2*dim, n_substeps+1, n_par)
	# Tr{σx*Reρ_i}...
	# Trick,  σx =(0  1)
	#             (1  0)... ..the trace effectively sums anti-diagonal terms
	# Reρ =  (     st. unimportant          ψRe[1]ψRe[2]+ψIm[1]ψIm[2])
	#	     (ψRe[1]ψRe[2]+ψIm[1]ψIm[2]        st. unimportant       )
	ψRe = @view sol[1:MyParameters.dim,:,:]   #(dim,n_substeps+1, n_par)
	ψIm = @view sol[MyParameters.dim+1:end,:,:]
	ex_x = 2*(ψRe[1,:,:].*ψRe[2,:,:]+ψIm[1,:,:].*ψIm[2,:,:]) #size(n_substeps+1, n_par)
end

function loss_along_trajectory(p1)
	#initial values
	loss = 0.0f0
	u = u0 #initial state
	#prepare ini Jhom, suppose there is no noise in the input
	ex_x = transpose(compute_ex_x(u0)) # (1,n_par)
	Jhom = reshape(repeat(ex_x,MyParameters.n_substeps),(MyParameters.n_substeps,n_par))
	αin = zeros(Float32,nαin,n_par)  #initially all past actions set to zero

	function prob_func(prob, i, repeat)
		#prepare tge vector of Wiener Process
		NG = CreateGrid(W1j[:,i]) # EM and RKMil
		remake(prob,p=α[i:i],u0=u[:,i],noise=NG)
	end

	#initial state, save the features
	Re_uj = @view u[1:MyParameters.dim,:]
	Im_uj= @view u[MyParameters.dim+1:end,:]
	fid = zeros(Float32,1,n_par)
	fid += abs2.(sum(Re_ut.*Re_uj,dims=1))
#	fid += abs2.(sum(Im_ut.*Im_uj,dims=1)) #commented because Im_ut=0, for general target uncomment
#	fid += abs2.(sum(Im_ut.*Re_uj,dims=1))
	fid += abs2.(sum(Re_ut.*Im_uj,dims=1))
#	fid += 2*(sum(Re_ut.*Re_uj,dims=1)).*sum(Im_ut.*Im_uj,dims=1)
#	fid -= 2*(sum(Re_ut.*Im_uj,dims=1)).*sum(Im_ut.*Re_uj,dims=1)
	mut_row(single_traj_fid_store,1,fid)
	mut(mean_fid_store,1,mean(fid))
	mut(std_fid_store,1,std(fid))

	W = sqrt(MyParameters.dt)*randn(Float32,MyParameters.n_substeps*MyParameters.n_steps+1,n_par) #for 1 trajectory
	W1 = cumsum([zero(MyParameters.dt); W[1:end-1,:]], dims=1)

	for j in 1:MyParameters.n_steps
		global W1j #must be global to be seen in prob_func
		global α
		#update action from NN
		α = re(p1)([Jhom,αin]).*MyParameters.force_mag #dim (1,n_par)
		αin = vcat(αin[2:end,:],α)
		#define ensemble problem
		W1j = @view W1[1+(j-1)*MyParameters.n_substeps:j*MyParameters.n_substeps+1,:]
		ensembleprob = EnsembleProblem(prob,prob_func = prob_func)

		sol = Array(solve(ensembleprob, RKMil(),
		ensemblealg=EnsembleThreads(), # EnsembleCPUArray(), EnsembleDistributed()
		sensealg = ForwardDiffSensitivity(),
		dt=MyParameters.dt, save_everystep=false,
		saveat=[t_interval],
		callback=cb, adaptive=false, trajectories=n_par, batch_size=n_par,
		save_noise=false,
		timeseries_errors=false,weak_timeseries_errors=false,weak_dense_errors=false
		))  #[2*dim,n_substeps+1, n_par]

		#collect <x> for all n_substeps and n_par
		ex_x = compute_ex_x(sol)  # (n_substeps, n_par), leave out the ini state
		dWj = W1j[2:end,:]-W1j[1:end-1,:]
		#NOISY MEASUREMENT at inidividual substeps
		Jhom = ex_x.*MyParameters.dt+dWj
		#current state
		u = @view sol[:,end,:]
		#compute loss function
		Re_uj = @view u[1:MyParameters.dim,:]
		Im_uj = @view u[MyParameters.dim+1:end,:]
		fid = zeros(Float32,1,n_par)
		fid += abs2.(sum(Re_ut.*Re_uj,dims=1))
	#	fid += abs2.(sum(Im_ut.*Im_uj,dims=1))
	#	fid += abs2.(sum(Im_ut.*Re_uj,dims=1))
		fid += abs2.(sum(Re_ut.*Im_uj,dims=1))
	#	fid += 2*(sum(Re_ut.*Re_uj,dims=1)).*sum(Im_ut.*Im_uj,dims=1)
	#	fid -= 2*(sum(Re_ut.*Im_uj,dims=1)).*sum(Im_ut.*Re_uj,dims=1)
		loss += C1*MyParameters.gamma^j*(1-mean(fid))
		mut(mean_fid_store,j+1,mean(fid))
		mut(std_fid_store,j+1,std(fid))
		mut_row(single_traj_fid_store,j+1,fid)
		mut(mean_action_store,j,mean(α))
		mut(std_action_store,j,std(α))
		mut_row(single_traj_action_store,j,α)
		#punish large actions--note, that for j we are pointing to the j-1st action!
		loss += C2*MyParameters.gamma^(j)*(mean(abs2.(α))) #mimic...sum max valus
   		#emphasize the main interval
   		if j>(MyParameters.n_steps-50)
  			loss += C3*MyParameters.gamma^j*(1-mean(fid))
   			mut_vec(Fock_end_example,Re_uj[:,1].^2+Im_uj[:,1].^2);
   		end
	end
	return loss
end


###################################
# training loop parameters
epochs=7000
println("total epochs: ",epochs)
training=zeros(epochs+1) #add extra dimension because of grad rejection condition in the training
maxgrads=zeros(epochs+1)
some_nans=zeros(epochs+1)

data = Iterators.repeated((), epochs)
opt = ADAM(0.0001)

#grad clipping
function clip(x)
    min.(max.(x,-40), 40)
end

using DelimitedFiles  #save txt files

## Save model
using BSON: @save
function qb_train!(loss, p1, data, opt,u0)
	ps = Flux.params(p1)
	iter = 1  #start with iter=1 because of the grad rejection lines below
	for d in data
		iter += 1
		prepare_initial!(u0)  #different initial states!
		#ini_fid=abs2.(sum(Re_ut.*u0[1:2],dims=1))+abs2.(sum(Re_ut.*u0[3:4],dims=1)) #initial mean fidelity
		@show iter
		#@show mean(ini_fid)
		@time gs = gradient(ps) do
			training_loss = loss(Zygote.hook(clip,p1)) #grad clipping does not work now???
			mut(training,iter,training_loss)
			println("loss: ",training_loss)
			return training_loss
		end
		maxgrads[iter]=maximum(abs.(gs[p1]))
		some_nans[iter]=sum(isnan.(gs[p1]))
		println("is nan: ",sum(isnan.(gs[p1])))
		println("max grad: ",maximum(abs.(gs[p1])))
		if iter%1 == 0
			fig1 = plot( [1:MyParameters.n_steps+1,1:MyParameters.n_steps+1],
			  [mean_fid_store mean_fid_store],
			    fillrange=[mean_fid_store.-std_fid_store mean_fid_store.+std_fid_store], fillalpha=0.3, c=:blue,
				xlabel = "steps", ylim=(0,1),xlim=(0,150), lw = 1.5, title="Fidelity" ,legend=false)
			fig1=plot!(1:MyParameters.n_steps+1,single_traj_fid_store[:,1], c=:black  )
			fig2 = plot( [1:MyParameters.n_steps,1:MyParameters.n_steps],
			  [mean_action_store mean_action_store],
			    fillrange=[mean_action_store.-std_action_store mean_action_store.+std_action_store], fillalpha=0.3, c=:orange,
				xlabel = "steps", ylim=(-MyParameters.force_mag,MyParameters.force_mag), xlim=(0,150), lw = 1.5, title="Action" ,legend=false)
			fig2=plot!(1:MyParameters.n_steps,single_traj_action_store[:,1], c=:black  )
			#display(plot(fig1, fig2, layout = (1, 2), legend = false,size=(800,300)));
			if iter == 1 || iter%100 == 0
				png(string("Figures/Figure_",iter)) #for saving figs
				epoch_output=hcat(1:MyParameters.n_steps,mean_fid_store[1:end-1],mean_action_store) #save results for epoch
				writedlm(string("Data/Epoch",iter,".txt"), epoch_output)
			end
			println("iter: ", iter)
		end
		println("++++++++++++++++++++++")
		# Only gradients that don't lead to a rapid increase of the loss function are accepted
		# Empirically it prevents instabilities in learning
		if training[iter]<1.1* training[iter-1]
			Flux.Optimise.update!(opt, ps, gs)
		else
			println("GRADS REJECTED!")
		end
		if iter%1000 == 0 #save the model every 1000 epochs
			@save string("model_Jhom-in_parallel_",iter,".bson")  p1 re opt
		end
	end
	@save string("Data/model_Jhom-in_parallel_END.bson")  p1 re opt
end

#training
Random.seed!(50)
qb_train!(loss_along_trajectory, p1, data, opt,u0)

#plotting of the loss function
plot(training[2:end], xlabel = "epochs", ylabel="loss", lw = 1.5, title = "Training")
savefig("Figures/Loss.pdf")
#save training a txt
writedlm("Data/Loss.txt", training[2:end])
