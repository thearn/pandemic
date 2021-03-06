import numpy as np
import openmdao.api as om

from pandemic.models.ks import KS
from pandemic.models.SEIR import SEIR
from pandemic.bootstrap_problem import generate_phase, make_plots, setup_and_run_phase

# ============== Default configuration =============

pop_total = 1.0
initial_exposure = 0.01 * pop_total
# model discretization 
ns = 50
# defect scaler for model solution
ds = 1e-1

#### VECTOR PARAMS
# baseline contact rate (infectivity)
beta = 0.25
# recovery rate (1/days needed to recover)
gamma = 1.0 / 14.0
# incubation rate (1/days needed to become infectious)
alpha = 1.0 / 5.0
# immunity loss rate (1/days needed to become susceptible again)
epsilon = 1.0 / 300.0

# set up model states
states = {'S' : {'name' : 'susceptible', 'rate_source' : 'Sdot', 
                 'targets' : ['S'], 'defect_scaler' : ds, 
                 'interp_s' : pop_total - initial_exposure, 'interp_f' : 0, 'c' : 'orange'},
          'E' : {'name' : 'exposed', 'rate_source' : 'Edot', 
                 'targets' : ['E'], 'defect_scaler' : ds, 
                 'interp_s' : initial_exposure, 'interp_f' : 0.0, 'c' : 'brown'},
          'I' : {'name' : 'infected', 'rate_source' : 'Idot', 
                 'targets' : ['I'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'green'},
                 }

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma},
          'alpha' : {'targets' : ['alpha'], 'val' : alpha},
          'epsilon' : {'targets' : ['epsilon'], 'val' : epsilon}}

# set up model scalar params
s_params = {'t_on' : {'targets' : ['t_on'], 'val' : 10.0},
            't_off' : {'targets' : ['t_off'], 'val' : 80.0},
            'a' : {'targets' : ['a'], 'val' : 5.0}}


class SEIRS(SEIR):
    """SEIRS epidemiological infection model
       S (suceptible), E (exposed), I (infected), R (recovered)

       This model includes immunity loss from states R to S at rate 'epsilon'.
    """
    def setup(self):
        super(SEIRS, self).setup()
        # inherits S, E, I, R, and all params

        nn = self.options['num_nodes']

        # New params
        self.add_input('epsilon',
               val = np.zeros(nn))

        arange = np.arange(self.options['num_nodes'], dtype=int)

        # append to previously defined partials
        self.declare_partials('Sdot', ['epsilon', 'R'], rows=arange, cols=arange)
        self.declare_partials('Rdot', ['epsilon', 'R'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        super(SEIRS, self).compute(inputs, outputs)

        S = inputs['S']
        E = inputs['E']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        epsilon = inputs['epsilon']
        theta = self.theta

        outputs['Sdot'] = - theta * S * I + epsilon * R
        outputs['Rdot'] = gamma * I - epsilon * R

    def compute_partials(self, inputs, jacobian):
        super(SEIRS, self).compute_partials(inputs, jacobian)

        S = inputs['S']
        E = inputs['E']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        epsilon = inputs['epsilon']
        theta = self.theta

        # derivatives of the ODE equations of state
        jacobian['Sdot', 'epsilon'] = R
        jacobian['Sdot', 'R'] = epsilon

        jacobian['Rdot', 'epsilon'] = - R
        jacobian['Rdot', 'R'] = - epsilon

if __name__ == '__main__':
    import dymos as dm
    import matplotlib.pyplot as plt

    # test derivatives
    p = om.Problem()
    p.model = om.Group()
    n = 35
    p.model.add_subsystem('test', SEIRS(num_nodes=n, truncate=False), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)
    p['S'] = np.random.uniform(1, 1000, n)
    p['E'] = np.random.uniform(1, 1000, n)
    p['I'] = np.random.uniform(1, 1000, n)
    p['R'] = np.random.uniform(1, 1000, n)

    p['beta'] = np.random.uniform(0, 2, n)
    p['sigma'] = np.random.uniform(0, 2, n)
    p['gamma'] = np.random.uniform(0, 2, n)
    p['alpha'] = np.random.uniform(0, 2, n)
    p['t'] = np.linspace(0, 100, n)
    
    p.run_model()
    p.check_partials(compact_print=True, method='cs')

    print()
    raw = input("Continue with baseline sim run test? (y/n)")
    if raw != "y":
        quit()

    p, phase0, traj = generate_phase(SEIRS, ns, states, params, s_params, t_initial_bounds, t_duration_bounds, fix_initial=True)


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

    make_plots(states, params)

    plt.show()
