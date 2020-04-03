from sympy import *

# type in component param names here
inputs = 'beta gamma susceptible infected immune dead N epsilon delta duration_infection duration_immune'

# ----------------
outputs = {}
inputs_unpacked = ', '.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------

# paste compute() code here

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

# ------------------
# ------------------
print()
print("    def compute_partials(self, inputs, partials):\n")
inputs_ns = ', '.join(["inputs['%s']" % inp for inp in inputs.split()])
print("       ", inputs_unpacked, "=", inputs_ns )
declare = {}

for oname in outputs:
    print()
    declare[oname] = []
    for iname in input_symbs:
        deriv = diff(outputs[oname], iname)
        if deriv != 0:
            if deriv == 1:
                deriv = 1.0

            deriv = 'np.exp'.join(str(deriv).split('exp'))
            st = "\t\tjacobian['%s', '%s'] = %s" % (oname, iname, deriv)
            print(st)
            declare[oname].append(iname)

# declare partials
# ------------------
print("")
print('\t\t' + 20*'#')
print("\t\tarange = np.arange(self.options['num_nodes'], dtype=int)")
for oname in declare:
    list_inputs = ["'" + str(i) + "'" for i in declare[oname]]
    list_inputs = ', '.join(list_inputs)
    declare_statements = "\t\tself.declare_partials('%s', [%s], rows=arange, cols=arange)" % (oname, list_inputs)
    print(declare_statements)

