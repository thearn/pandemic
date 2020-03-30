import numpy as np
import openmdao.api as om

from infection import Infection
from trigger_comp import EventTrigger

class Pandemic(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('beta_trigger', EventTrigger(num_nodes=nn))

        self.add_subsystem('infection', Infection(num_nodes=nn), 
                           promotes_inputs=['susceptible', 
                                            'infected',
                                            'immune',
                                            'dead',
                                            'critical_unhospitalized',
                                            'critical_hospitalized',
                                            'duration_infection',
                                            'duration_immune',
                                            'duration_critical',
                                            'epsilon',
                                            'gamma',
                                            'mu',
                                            'sigma',
                                            'tau'],
                           promotes_outputs=['sdot', 'idot', 'rdot', 'ddot',
                                             'chdot', 'cudot', 'N'])

        self.linear_solver = om.DirectSolver()

        self.connect('beta_trigger.filtered', 'infection.beta')