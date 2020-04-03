from graphviz import Digraph

dot = Digraph(comment='COVID_mdao', format='png')
dot.attr(size='5000,5000')


dot.node('S', 'Susceptible')
dot.node('E', 'Exposed')
dot.node('I', 'Infected')
dot.node('R', 'Recovered')
dot.node('D', 'Dead')

dot.edge('S', 'E', label="(Incubation for 5d)")
dot.edge('E', 'I', label="(Becomes contagious for 14d)")

dot.edge('I', 'R', label="(Recovers with immunity 95%)")
dot.edge('I', 'D', label="(Dies 5%)")

dot.edge('R', 'S', label="(Loses immunity 300d)")

dot.render('covid_mdao', view=True)