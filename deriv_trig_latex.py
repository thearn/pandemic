from sympy import *

# type in component param names here
inputs = 't t_on t_off sigma a beta'

# ----------------
outputs = {}
inputs_unpacked = ', '.join(inputs.split())
exec('%s = symbols("%s")' % (inputs_unpacked, inputs))
exec('input_symbs = [%s]' % inputs_unpacked)
# -----------------
# -----------------

y = 1 / (1 + exp(-a*(t - t_on))) * 1 / (1 + exp(-a*(t_off - t))) 

filtered = sigma*y + (1 - y) * beta


print(latex(filtered.simplify()))


quit()



filtered = signal*y + (1 - y) * default_val

outputs['filtered'] = filtered
outputs['filtered_timescaled'] = (default_val - signal)**2


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


# run the file to get compute_partials code