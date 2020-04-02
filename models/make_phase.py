import numpy as np
import openmdao.api as om
from SIR_no_bool import SIR
import dymos as dm
import matplotlib.pyplot as plt


def make_phase(ODEclass, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True):
    phase = dm.Phase(ode_class=ODEclass,
                   transcription=dm.GaussLobatto(num_segments=ns, 
                                                 order=3))
    traj.add_phase(name='phase0', phase=phase)

    phase.set_time_options(initial_bounds=t_initial_bounds,
                           duration_bounds=t_duration_bounds, 
                           targets=['t'])


    for state in states:
        sdata = states[state]

        phase.add_state(state, 
                        fix_initial=fix_initial, 
                        rate_source=sdata['rate_source'], 
                        targets=sdata['targets'], 
                        lower=0.0,
                        upper=pop_total, 
                        ref=pop_total/2, 
                        defect_scaler = sdata['defect_scaler'])

    for param in params:
        pdata = params[param]
        phase.add_input_parameter(param, 
                                  targets=pdata['targets'], 
                                  dynamic=True, 
                                  val=pdata['val'])

    for param in s_params:
        pdata = s_params[param]
        phase.add_input_parameter(param, 
                                  targets=pdata['targets'], 
                                  dynamic=False, 
                                  val=pdata['val'])
    return phase

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
# recovery rate (1/days needed to recover)
gamma = 1.0 / 14.0

# set up model states
states = {'S' : {'name' : 'susceptible', 'rate_source' : 'Sdot', 
                 'targets' : ['S'], 'defect_scaler' : ds, 
                 'interp_s' : pop_total - initial_exposure, 'interp_f' : 0, 'c' : 'orange'},
          'I' : {'name' : 'infected', 'rate_source' : 'Idot', 
                 'targets' : ['I'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'green'},
                 }

t_duration_bounds = [5.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma}}

# set up model scalar params
s_params = {}

p = om.Problem(model=om.Group())
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)
p.model.linear_solver = om.DirectSolver()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
#p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
#p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['iSumm'] = 6

############################################

phase0 = make_phase(SIR, states, params, s_params, [0.0, 1.0], [1.0, 30.0], fix_initial=True)

phase0.add_boundary_constraint('I', loc='final', equals=0.1)
phase0.add_objective('time', loc='final')
phase0.add_timeseries_output('theta')
traj.add_phase(name='phase0', phase=phase0)



phase1 = make_phase(SIR, states, params, s_params, [1.0, 30.0], [60.0, 61.0], fix_initial=False)
phase1.add_control('sigma', targets=['sigma'], lower=0.0, upper=beta, ref=beta)
phase1.add_objective('max_I', scaler=1e3)
phase1.add_timeseries_output('theta')
traj.add_phase(name='phase1', phase=phase1)


phase2 = make_phase(SIR, states, params, s_params, [50, 200],  [1.0, 300.0], fix_initial=False)
phase2.add_objective('time', loc='final')
phase2.add_boundary_constraint('I', loc='final', upper=0.01)
phase2.add_timeseries_output('theta')
traj.add_phase(name='phase2', phase=phase2)


traj.link_phases(phases=['phase0', 'phase1', 'phase2'],
                 vars=['time', 'S', 'I', 'R'])

p.driver.declare_coloring() 
p.setup(check=True)

quit()



############################################

p.driver.declare_coloring() 
p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', np.mean(t_duration_bounds))

for state in states:
    sdata = states[state]
    p.set_val('traj.phase0.states:%s' % state,
              phase.interpolate(ys=[sdata['interp_s'], 
                                    sdata['interp_f']], 
                                nodes='state_input'))


p.run_driver()
sim_out = traj.simulate()

t = sim_out.get_val('traj.phase0.timeseries.time')
theta = sim_out.get_val('traj.phase0.timeseries.theta')

for state in states:
    states[state]['result'] = sim_out.get_val('traj.phase0.timeseries.states:%s' % state)


############################################

fig = plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.title('baseline simulation - no mitigation')
for state in states:
    plt.plot(t, states[state]['result'], lw=2, label=states[state]['name'])
plt.xlabel('days')
plt.legend()

plt.subplot(212)
plt.plot(t, len(t)*[beta], lw=2, label='$\\beta$')
plt.plot(t, theta, lw=2, label='$\\theta$(t)')
plt.legend()
plt.show()