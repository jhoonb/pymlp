###############################################
# Test Multilayer Perceptron
# Jhonathan Paulo Banczek - 2013
# jpbanczek@gmail.com github.com/jhoonb/pymlp
# Problem XOR 
###############################################

from pymlp import Pymlp


mlp = Pymlp((2,5,1))

conf = {}

conf['input'] = [[0,0],[0,1],[1,0],[1,1]]
conf['output'] = [[0],[1],[1],[0]]
conf['it'] = 10000
conf['lr'] = 0.5
conf['bias'] = 1
conf['mm'] = 1
conf['error'] = 0.010
conf['mode'] = 'sequential'
conf['fx'] = None
conf['dfx'] = None

mlp.configuration(conf)
mlp.training()
mlp.test(conf['input'])
