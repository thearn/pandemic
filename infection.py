import numpy as np
from openmdao.api import ExplicitComponent


class Infection(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # States
        self.add_input('susceptible',
                       val=np.zeros(nn),
                       desc='susceptible',
                       units='pax')

        self.add_input('infected',
                       val=np.zeros(nn),
                       desc='infected',
                       units='pax')

        self.add_input('immune',
                       val=np.zeros(nn),
                       desc='immune',
                       units='pax')

        self.add_input('dead',
                       val=np.zeros(nn),
                       desc='dead',
                       units='pax')

        self.add_input('critical_hospitalized',
                       val=np.zeros(nn),
                       desc='critical hospitalized')

        self.add_input('critical_unhospitalized',
                       val=np.zeros(nn),
                       desc='critical_unhospitalized')

        # ROCs
        self.add_output('sdot', val=np.zeros(nn), units='pax/d')
        self.add_output('idot', val=np.zeros(nn), units='pax/d')
        self.add_output('rdot', val=np.zeros(nn), units='pax/d')
        self.add_output('ddot', val=np.zeros(nn), units='pax/d')
        self.add_output('chdot', val=np.zeros(nn), units='pax/d')
        self.add_output('cudot', val=np.zeros(nn), units='pax/d')

        self.add_output('beta_pass', val=np.zeros(nn))

        self.add_output('N',
                       val=np.zeros(nn),
                       desc='population total', units='pax')



        self.add_input('t',
                       val=np.zeros(nn),
                       desc='time',
                       units='d')

        # self.add_output('hdot', val=np.zeros(nn), units='1.0/d')
        # self.add_output('cdot', val=np.zeros(nn), units='1.0/d')

        # Params
        self.add_input('epsilon',
                       val=0.0 * np.ones(nn), desc='immunity loss rate', units=None)

        self.add_input('beta',
                       val=0.4 * np.ones(nn), desc='contact rate', units=None)

        self.add_input('gamma',
                       val=0.9 * np.ones(nn), desc='recovery rate', units=None)

        self.add_input('mu',
                       val=0.2 * np.ones(nn), desc='critical rate', units=None)

        self.add_input('sigma',
                       val=0.05 * np.ones(nn), desc='death rate critical hospitalized', units=None)

        self.add_input('tau',
                       val=0.9 * np.ones(nn), desc='death rate critical, unhospitalized', units=None)

        # durations
        self.add_input('duration_infection',
                       val=14. * np.ones(nn), desc='duration of the infection', units='d')

        self.add_input('duration_immune',
                       val=300.0 * np.ones(nn), desc='duration of immunity', units='d')
        
        self.add_input('duration_critical',
                       val=1.e10 * np.ones(nn), desc='duration of immunity', units='d')

        arange = np.arange(self.options['num_nodes'], dtype=int)
        self.declare_partials('sdot', ['beta', 'susceptible', 'infected', 'immune', 'dead', 'epsilon', 'duration_immune'], rows=arange, cols=arange)
        self.declare_partials('idot', ['beta', 'susceptible', 'infected', 'immune', 'dead', 'duration_infection'], rows=arange, cols=arange)
        self.declare_partials('rdot', ['gamma', 'infected', 'immune', 'epsilon', 'duration_infection', 'duration_immune'], rows=arange, cols=arange)
        self.declare_partials('ddot', ['gamma', 'infected', 'duration_infection'], rows=arange, cols=arange)
        self.declare_partials('N', ['susceptible', 'infected', 'immune', 'dead'], rows=arange, cols=arange)
        self.declare_partials('beta_pass', ['beta'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        beta, gamma, susceptible, infected, immune, dead, epsilon, duration_infection, duration_immune = inputs['beta'], inputs['gamma'], inputs['susceptible'], inputs['infected'], inputs['immune'], inputs['dead'], inputs['epsilon'], inputs['duration_infection'], inputs['duration_immune']

        epsilon = inputs['epsilon']

        N = susceptible + infected + immune + dead
        pct_infected = infected / N
        
        new_infected = susceptible * beta * pct_infected

        new_recovered = infected * gamma/duration_infection
        
        new_susceptible = immune * epsilon / duration_immune
        
        new_dead = infected * (1 - gamma) / duration_infection

        outputs['sdot'] = new_susceptible - new_infected

        outputs['idot'] = new_infected - new_recovered - new_dead

        outputs['rdot'] = new_recovered - new_susceptible

        outputs['ddot'] = new_dead

        outputs['N'] = N

        outputs['beta_pass'] = beta


    def compute_partials(self, inputs, jacobian):
        beta, gamma, susceptible, infected, immune, dead, epsilon, duration_infection, duration_immune = inputs['beta'], inputs['gamma'], inputs['susceptible'], inputs['infected'], inputs['immune'], inputs['dead'], inputs['epsilon'], inputs['duration_infection'], inputs['duration_immune']

        jacobian['sdot', 'beta'] = -infected*susceptible/(dead + immune + infected + susceptible)
        jacobian['sdot', 'susceptible'] = beta*infected*susceptible/(dead + immune + infected + susceptible)**2 - beta*infected/(dead + immune + infected + susceptible)
        jacobian['sdot', 'infected'] = beta*infected*susceptible/(dead + immune + infected + susceptible)**2 - beta*susceptible/(dead + immune + infected + susceptible)
        jacobian['sdot', 'immune'] = beta*infected*susceptible/(dead + immune + infected + susceptible)**2 + epsilon/duration_immune
        jacobian['sdot', 'dead'] = beta*infected*susceptible/(dead + immune + infected + susceptible)**2
        jacobian['sdot', 'epsilon'] = immune/duration_immune
        jacobian['sdot', 'duration_immune'] = -epsilon*immune/duration_immune**2

        jacobian['idot', 'beta'] = infected*susceptible/(dead + immune + infected + susceptible)
        jacobian['idot', 'susceptible'] = -beta*infected*susceptible/(dead + immune + infected + susceptible)**2 + beta*infected/(dead + immune + infected + susceptible)
        jacobian['idot', 'infected'] = -beta*infected*susceptible/(dead + immune + infected + susceptible)**2 + beta*susceptible/(dead + immune + infected + susceptible) - gamma/duration_infection - (1 - gamma)/duration_infection
        jacobian['idot', 'immune'] = -beta*infected*susceptible/(dead + immune + infected + susceptible)**2
        jacobian['idot', 'dead'] = -beta*infected*susceptible/(dead + immune + infected + susceptible)**2
        jacobian['idot', 'duration_infection'] = gamma*infected/duration_infection**2 + infected*(1 - gamma)/duration_infection**2

        jacobian['rdot', 'gamma'] = infected/duration_infection
        jacobian['rdot', 'infected'] = gamma/duration_infection
        jacobian['rdot', 'immune'] = -epsilon/duration_immune
        jacobian['rdot', 'epsilon'] = -immune/duration_immune
        jacobian['rdot', 'duration_infection'] = -gamma*infected/duration_infection**2
        jacobian['rdot', 'duration_immune'] = epsilon*immune/duration_immune**2

        jacobian['ddot', 'gamma'] = -infected/duration_infection
        jacobian['ddot', 'infected'] = (1 - gamma)/duration_infection
        jacobian['ddot', 'duration_infection'] = -infected*(1 - gamma)/duration_infection**2

        jacobian['N', 'susceptible'] = 1.0
        jacobian['N', 'infected'] = 1.0
        jacobian['N', 'immune'] = 1.0
        jacobian['N', 'dead'] = 1.0

        jacobian['beta_pass', 'beta'] = 1.0

if __name__ == '__main__':
    from openmdao.api import Problem, Group

    p = Problem()
    p.model = Group()

    n = 35
    p.model.add_subsystem('test', Infection(num_nodes=n), promotes=['*'])
    p.setup(force_alloc_complex=True)
    np.random.seed(0)

    p['susceptible'] = np.random.uniform(1, 1000, n)
    p['infected'] = np.random.uniform(1, 1000, n)
    p['immune'] = np.random.uniform(1, 1000, n)
    p['dead'] = np.random.uniform(1, 1000, n)

    p.run_model()


    p.check_partials(compact_print=True, method='cs')


