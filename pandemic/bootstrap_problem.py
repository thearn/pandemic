import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt


def generate_phase(ODEclass, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True, fix_duration=False):
    p = om.Problem(model=om.Group())
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)
    p.model.linear_solver = om.DirectSolver()

    phase = dm.Phase(ode_class=ODEclass,
                   transcription=dm.GaussLobatto(num_segments=ns, 
                                                 order=3))
    traj.add_phase(name='phase0', phase=phase)


    if fix_duration:
        phase.set_time_options(fix_initial=fix_initial, fix_duration=True, targets=['t'], units='d')

    else:
        phase.set_time_options(initial_bounds=t_initial_bounds,
                               duration_bounds=t_duration_bounds, 
                               targets=['t'], units='d')

    traj.add_phase(name='phase0', phase=phase)

    for state in states:
        sdata = states[state]

        phase.add_state(state, 
                        fix_initial=fix_initial, 
                        rate_source=sdata['rate_source'], 
                        targets=sdata['targets'], 
                        lower=0.0,
                        upper=1.0, 
                        ref=0.5, 
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
    return p, phase, traj

def setup_and_run_phase(states, p, phase0, traj, t_duration):
    p.driver.declare_coloring() 
    p.setup(check=True)
    p.set_val('traj.phase0.t_initial', 0)
    p.set_val('traj.phase0.t_duration', t_duration)

    for state in states:
        sdata = states[state]
        p.set_val('traj.phase0.states:%s' % state,
                  phase0.interpolate(ys=[sdata['interp_s'], 
                                        sdata['interp_f']], 
                                    nodes='state_input'))


    p.run_driver()
    sim_out = p#traj.simulate()

    t = sim_out.get_val('traj.phase0.timeseries.time')
    theta = sim_out.get_val('traj.phase0.timeseries.theta')

    for state in states:
        states[state]['result'] = sim_out.get_val('traj.phase0.timeseries.states:%s' % state)
    states['t'] = t
    states['theta'] = theta

def make_plots(states, params, ignore = ['t', 'theta', 'int_sigma']):
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(211)
    t = states['t']
    beta = params['beta']['val']
    theta = states['theta']
    for state in states:
        if state in ignore:
            continue
        plt.plot(t, states[state]['result'], states[state]['c'], lw=2, label=states[state]['name'])
    plt.xlabel('days')
    plt.legend()

    plt.subplot(212)
    plt.plot(t, len(t)*[beta], lw=2, label='$\\beta$')
    plt.plot(t, theta, lw=2, label='$\\theta$(t)')
    plt.legend()
    #plt.show()
    return fig



