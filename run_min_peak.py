import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver

from pandemic import Pandemic

pop_total = 1.0 * 1e6
infected0 = int(0.0005 * pop_total) + 1
ns = 35

p = om.Problem(model=om.Group())

traj = dm.Trajectory()

p.model.add_subsystem('traj', subsys=traj)

phase = dm.Phase(ode_class=Pandemic,
                 transcription=dm.GaussLobatto(num_segments=ns, 
                                               order=3))
p.model.linear_solver = DirectSolver()
#phase.set_time_options(fix_initial=True, duration_bounds=(200.0, 301.0), units='d', targets=['sigma_comp.t'])
phase.set_time_options(fix_initial=True, fix_duration=True, units='d', targets=['sigma_comp.t'])


ds = 1e-3
phase.add_state('susceptible', fix_initial=True, units='pax', rate_source='sdot', targets=['susceptible'],
                ref=pop_total/2.0, defect_scaler = ds)
phase.add_state('dead', fix_initial=True, units='pax', rate_source='ddot', targets=['dead'],
                ref=pop_total/2.0, defect_scaler = ds)
phase.add_state('infected', fix_initial=True, units='pax', rate_source='idot', targets=['infected'],
                ref=pop_total/2.0, defect_scaler = ds)
phase.add_state('immune', fix_initial=True, units='pax', rate_source='rdot', targets=['immune'],
                ref=pop_total/2.0, defect_scaler = ds)

#p.driver = om.ScipyOptimizeDriver()

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
#p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-8
p.driver.opt_settings['Major optimality tolerance'] = 1.0
p.driver.opt_settings["Major step limit"] = 0.5 #2.0
p.driver.opt_settings['Major iterations limit'] = 1000000
p.driver.opt_settings['Minor iterations limit'] = 1000000
p.driver.opt_settings['Iterations limit'] = 1000000
p.driver.opt_settings['iSumm'] = 6

p.driver.declare_coloring()


t_on = 15.0 
t_off = 55.0

phase.add_input_parameter('t_on', units='d', targets=['sigma_comp.t_on'], dynamic=False, val=t_on)
phase.add_input_parameter('t_off', units='d', targets=['sigma_comp.t_off'], dynamic=False, val=t_off)


lim = 0.95

phase.add_polynomial_control('sigma', targets=['sigma_comp.signal'], lower=0.05, upper=0.4, order=3)

phase.add_objective('max_infected', ref=1e2)

phase.add_timeseries_output('sigma_comp.filtered')



traj.add_phase(name='phase0', phase=phase)
p.setup(check=True)

p.set_val('traj.phase0.t_initial', 0)
p.set_val('traj.phase0.t_duration', 90)

p.set_val('traj.phase0.states:susceptible',
          phase.interpolate(ys=[pop_total - infected0, 0], nodes='state_input'))
p.set_val('traj.phase0.states:infected',
          phase.interpolate(ys=[infected0, 0], nodes='state_input'))
p.set_val('traj.phase0.states:immune',
          phase.interpolate(ys=[0, pop_total/2], nodes='state_input'))
p.set_val('traj.phase0.states:dead',
          phase.interpolate(ys=[1, 0], nodes='state_input'))

p.run_driver()
sim_out = traj.simulate()

t = sim_out.get_val('traj.phase0.timeseries.time')
s = sim_out.get_val('traj.phase0.timeseries.states:susceptible')
i = sim_out.get_val('traj.phase0.timeseries.states:infected')
r = sim_out.get_val('traj.phase0.timeseries.states:immune')
d = sim_out.get_val('traj.phase0.timeseries.states:dead')



theta = sim_out.get_val('traj.phase0.timeseries.filtered')

try:
  sigma = sim_out.get_val('traj.phase0.timeseries.controls:sigma')
except:
  sigma = sim_out.get_val('traj.phase0.timeseries.polynomial_controls:sigma')


fig = plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.title('polynomial mitigation between days %2.2f and %2.2f' % (t_on, t_off))
plt.plot([t_on,t_on], [0, 1], 'k--', linewidth=0.9)
plt.plot([t_off,t_off], [0, 1], 'k--', linewidth=0.9)
plt.plot(t, i/pop_total, label='infected')
plt.plot(t, s/pop_total, label='susceptible')
plt.plot(t, r/pop_total, label='recovd/immune')
plt.plot(t, d/pop_total, label='dead')
plt.xlabel('days')
plt.ylabel('pct. pop')
plt.legend(loc=1)

plt.subplot(212)
plt.plot([t_on,t_on], [0, 1], 'k--', linewidth=0.9)
plt.plot([t_off,t_off], [0, 1], 'k--', linewidth=0.9)
plt.plot(t, theta, label='eff. contact rate $\\theta$')
plt.plot(t, sigma, label='control $\\sigma$')
plt.legend()
plt.show()