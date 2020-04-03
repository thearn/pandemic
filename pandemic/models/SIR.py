import numpy as np
import openmdao.api as om

from pandemic.models.ks import KS
from pandemic.models.base_infection import BaseInfection
from pandemic.bootstrap_problem import generate_phase, make_plots, setup_and_run_phase

# ============== Default configuration =============

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
                 'interp_s' : initial_exposure, 'interp_f' : pop_total/2, 'c' : 'navy'},
          'R' : {'name' : 'recovered', 'rate_source' : 'Rdot', 
                 'targets' : ['R'], 'defect_scaler' : ds, 
                 'interp_s' : 0.0, 'interp_f' : pop_total/2, 'c' : 'green'},
                 }

t_initial_bounds = [0.0, 1.0]
t_duration_bounds = [200.0, 301.00]

# set up model vector params
params = {'beta' : {'targets' : ['beta'], 'val' : beta},
          'gamma' : {'targets' : ['gamma'], 'val' : gamma}}

# set up model scalar params
s_params = {'t_on' : {'targets' : ['t_on'], 'val' : 10.0},
            't_off' : {'targets' : ['t_off'], 'val' : 70.0},
            'a' : {'targets' : ['a'], 'val' : 5.0}}


class SIR(BaseInfection):
    """Basic epidemiological infection model
       S (suceptible), I (infected), R (recovered).
    """
    def setup(self):
        super(SIR, self).setup()

        nn = self.options['num_nodes']

        # States
        self.add_input('S',
               val=np.zeros(nn))

        self.add_input('I',
               val=np.zeros(nn))

        self.add_input('R',
               val=np.zeros(nn))

        # ROCs
        self.add_output('Sdot', val=np.zeros(nn))
        self.add_output('Idot', val=np.zeros(nn))
        self.add_output('Rdot', val=np.zeros(nn))

        # Params

        self.add_input('gamma',
               val = np.zeros(nn))

        self.add_output('max_I', val=0.0)

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials('Sdot', ['beta', 'sigma', 'gamma', 'S', 'I', 'R', 't'], rows=arange, cols=arange)
        self.declare_partials('Sdot', ['a', 't_on', 't_off'])

        self.declare_partials('Idot', ['beta', 'sigma', 'gamma', 'S', 'I', 't'], rows=arange, cols=arange)
        self.declare_partials('Idot', ['a', 't_on', 't_off'])

        self.declare_partials('Rdot', ['gamma', 'I'], rows=arange, cols=arange)

        self.declare_partials('max_I', 'I')

    def compute(self, inputs, outputs):
        super(SIR, self).compute(inputs, outputs)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        theta = self.theta

        # time derivatives of the states of an SIR model
        # substitution dynamic control 'theta' for constant 'beta'.
        outputs['Sdot'] = - theta * S * I
        outputs['Idot'] = theta * S * I - gamma * I
        outputs['Rdot'] = gamma * I

        # aggregate the infection curve to determine peak
        agg_i, self.dagg_i = KS(I)
        outputs['max_I'] = np.sum(agg_i)

    def compute_partials(self, inputs, jacobian):
        super(SIR, self).compute_partials(inputs, jacobian)

        S = inputs['S']
        I = inputs['I']
        R = inputs['R']
        gamma = inputs['gamma']
        theta = self.theta

        # derivatives of the ODE equations of state
        jacobian['Sdot', 'S'] = -I * theta
        jacobian['Sdot', 'I'] = -S * theta
        # cascade the derivatives w.r.t. theta using the chain rule
        dSdot_dtheta = -I * S
        self.apply_theta_derivs('Sdot', dSdot_dtheta, jacobian)

        jacobian['Idot', 'gamma'] = -I
        jacobian['Idot', 'S'] = I * theta
        jacobian['Idot', 'I'] = S * theta - gamma
        dIdot_dtheta = I * S
        self.apply_theta_derivs('Idot', dIdot_dtheta, jacobian)

        jacobian['Rdot', 'gamma'] = I
        jacobian['Rdot', 'I'] = gamma

        jacobian['max_I', 'I'] = self.dagg_i
