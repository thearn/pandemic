
import numpy as np
from pandemic.bootstrap_problem import generate_phase, setup_and_run_phase, make_plots
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error



def test_model_partials():
    # test derivatives
    p = om.Problem()
    p.model = om.Group()
    n = 35
    from pandemic.models.SEIRDS import SEIRDS, states, params, s_params

    p.model.add_subsystem('test', SEIRDS(num_nodes=n, truncate=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['E'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)
    p['D'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)
    p['alpha'] = np.random.uniform(0, 2, n)
    p['epsilon'] = np.random.uniform(0, 2, n)
    p['mu'] = np.random.uniform(0, 2, n)
    p['t'] = np.linspace(0, 100, n)
    p.run_model()
    p.check_partials(compact_print=True, method='cs')

def test_baseline_run():
    from pandemic.models.SEIRDS import SEIRDS, states, params, s_params, ns, t_duration_bounds, t_initial_bounds
    # test baseline model
    p, phase0, traj = generate_phase(SEIRDS, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True)


    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = False

    p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
    # p.driver.opt_settings['mu_init'] = 1.0E-2
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['linear_solver'] = 'mumps'
    p.driver.opt_settings['max_iter'] = 500

    phase0.add_boundary_constraint('I', loc='final', upper=0.01, scaler=1.0)
    
    phase0.add_objective('time', loc='final', scaler=1.0)

    phase0.add_timeseries_output('theta')
    
    setup_and_run_phase(states, p, phase0, traj, t_duration_bounds[0])

    max_I = np.max(states['I']['result'])
    final_R = states['R']['result'][-1]
    final_S = states['S']['result'][-1]
    final_D = states['D']['result'][-1]

    print("max_I", max_I)
    print("final_R", final_R)
    print("final_S", final_S)
    print("final_D", final_D)

    mI_actual = 0.26364939220443845
    fR_actual = 0.66815968
    fS_actual = 0.26927833
    fD_actual = 0.05378587

    tol = 1e-2

    assert (abs(max_I - mI_actual) / mI_actual) < tol
    assert (abs(final_R - fR_actual) / fR_actual) < tol
    assert (abs(final_S - fS_actual) / fS_actual) < tol
    assert (abs(final_D - fD_actual) / fD_actual) < tol

    return states, params


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_model_partials()

    states, params = test_baseline_run()
    fig = make_plots(states, params)
    fig.suptitle('Baseline SEIRDS')

    plt.show()