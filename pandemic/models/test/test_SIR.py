
import numpy as np
from pandemic.bootstrap_problem import generate_phase, setup_and_run_phase, make_plots
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error



def test_model_partials():
    # test derivatives
    from pandemic.models.SIR import SIR, states, params, s_params

    np.random.seed(0)

    p = om.Problem()
    p.model = om.Group()
    
    n = 35
    p.model.add_subsystem('test', SIR(num_nodes=n, truncate=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)

    p['t'] = np.linspace(0, 100, n)
    
    p.run_model()

    x = p.check_partials(compact_print=True, method='cs')

    assert_check_partials(x)

def test_baseline_run():
    ns = 35
    t_initial_bounds = [0.0, 1.0]
    t_duration_bounds = [200.0, 301.00]
    from pandemic.models.SIR import SIR, states, params, s_params

    p, phase0, traj = generate_phase(SIR, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True)

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False
    p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    # p.driver.opt_settings['mu_init'] = 1.0E-2
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['linear_solver'] = 'mumps'
    p.driver.opt_settings['max_iter'] = 500

    phase0.add_boundary_constraint('I', loc='final', upper=0.05, scaler=1.0)
    
    phase0.add_objective('time', loc='final', scaler=1.0)

    phase0.add_timeseries_output('theta')
    
    setup_and_run_phase(states, p, phase0, traj, t_duration_bounds[0])

    max_I = np.max(states['I']['result'])
    final_S = states['S']['result'][-1]

    print("max_I", max_I)
    print("final_S", final_S)
    err = abs(max_I - 0.3592) / 0.3592
    assert (err < 1e4)

    return states, params

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_model_partials()
    
    states, params = test_baseline_run()

    fig = make_plots(states, params)
    fig.suptitle('Baseline SIR')
    plt.show()
