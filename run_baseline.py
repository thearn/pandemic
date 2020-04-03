import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om 

from models import SEIRDS # SIR, SEIR, SEIRS
from models.bootstrap_model import generate_phase, setup_and_run_phase, make_plots

# test baseline model
pop_total = 1.0
initial_exposure = 0.01 * pop_total
# model discretization 
ns = 50
# defect scaler for model solution
ds = 1e-2

#### VECTOR PARAMS
# baseline contact rate (infectivity)
beta = 0.25
# recovery rate (recovery rate / days needed to resolve infection)
gamma = 0.95 / 14.0
# incubation rate (1/days needed to become infectious)
alpha = 1.0 / 5.0
# immunity loss rate (1/days needed to become susceptible again)
epsilon = 1.0 / 300.0
# death rate (mortality rate / days average to resolve infection)
#   should be complementry to recovery (gamma)
mu = 0.05 / 14.0

# set up model states
states = {'S' : {'name' : 'susceptible', 'rate_source' : 'Sdot', 
                 'targets' : ['S'], 'defect_scaler' : ds, 
                 'interp_s' : pop_total - initial_exposure, 'interp_f' : 0, 'c' : 'orange'},
          'E' : {'name' : 'exposed', 'rate_source' : 'Edot', 
                 'targets' : ['E'], 'defect_scaler' : ds, 
                 'interp_s' : initial_exposure, 'interp_f' : 0.0, 'c' : 'brown'},
          'I' : {'name' : 'infected', 'rate_source' : 'Idot', 
                 'targets' : ['I'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/3, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/3, 'c' : 'green'},
          'D' : {'name' : 'died', 'rate_source' : 'Ddot', 
                 'targets' : ['D'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/3, 'c' : 'red'},
                 }

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma},
          'alpha' : {'targets' : ['alpha'], 'val' : alpha},
          'epsilon' : {'targets' : ['epsilon'], 'val' : epsilon},
          'mu' : {'targets' : ['mu'], 'val' : mu}}

p, phase0, traj = generate_phase(SEIRDS, ns, states, params, {}, 
                                 t_initial_bounds, t_duration_bounds, 
                                 fix_initial=True, fix_duration=True)


p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'IPOPT'
p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
# p.driver.opt_settings['mu_init'] = 1.0E-2
p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
p.driver.opt_settings['print_level'] = 5
p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.opt_settings['max_iter'] = 500

phase0.add_boundary_constraint('I', loc='final', upper=0.01, scaler=1.0)

#phase0.add_control('sigma', targets=['sigma'], lower=0.0, upper=beta, ref=beta)
#phase0.add_objective('max_I', scaler=1000.0)

phase0.add_objective('time', loc='final', scaler=1.0)

phase0.add_timeseries_output('theta')


setup_and_run_phase(states, p, phase0, traj, 200.0)

print(states['I']['result'][-1])
make_plots(states, params)