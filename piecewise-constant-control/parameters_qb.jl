# In this module, all parameters are defined:

module parameters
export parameters_qb

parameters_para=Dict(
    "N" =>2,  
    "w" => 20.0f0, 
    "force_mag" => 5.0f0,
    "max_episode_steps" => 150,   
    "dt" => 0.001f0,
    "n_substeps" => 20,
    "gamma" => 1.0f0)
end
