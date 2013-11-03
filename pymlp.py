###############################################
# Multilayer Perceptron in Python
# Jhonathan Paulo Banczek - 2013
# jpbanczek@gmail.com github.com/jhoonb/pymlp
###############################################

from random import random
from math import tanh

def narray(nl, nc, value):
    '''
     Create array bi-dimensional
     nl: number of lines
     nc: number of colunns
     value: 'r' = randomic values
    '''
    
    n = []

    if value is None:
        value = 0


    for i in range(nl):
        l = []
        for j in range(nc):
            if value == 'r':
                l.append(random() * 0.2)
            else:
                l.append(value)
        n.append(l)
    return n



def ftanh(x):
    ''' Default function activation: tangent hiperbolic '''

    return tanh(x)


def dftanh(x):
    ''' Default function derivate function activation '''
    return 1 - ((ftanh(x)) ** 2)


class Pymlp(object):
    ''' Pymlp Class Multilayer Perceptron'''

    def __init__(self, topology):

        self._topology = topology

        if len(self._topology) == 3:

            self._ninput = self._topology[0]
            self._nhidden1 = self._topology[1]
            self._noutput = self._topology[2]

            # array for sum in layers
            self._sum_h1 = [0] * self._nhidden1
            self._sum_out = [0] * self._noutput

            #array for output in layers
            self._y_in = [0] * self._ninput
            self._y_h1 = [0] * self._nhidden1
            self._y_out = [0] * self._noutput

            #create weight from layers
            self._w_in_h1 = narray(self._ninput, self._nhidden1, 'r')
            self._w_h1_out = narray(self._nhidden1, self._topology[2], 'r') 

            #create mommentum 
            self._mm_in_h1 = narray(self._ninput, self._nhidden1, 'r')
            self._mm_h1_out = narray(self._nhidden1, self._topology[2], 'r')

        elif len(self._topology) == 4:
            self._ninput = self._topology[0]
            self._nhidden1 = self._topology[1]
            self._nhidden2 = self._topology[2]
            self._noutput = self._topology[3]

            # array for sum in layers
            self._sum_h1 = [0] * self._nhidden1
            self._sum_h2 = [0] * self._nhidden2
            self._sum_out = [0] * self._noutput

            #array for output in layers
            self._y_in = [0] * self._ninput
            self._y_h1 = [0] * self._nhidden1
            self._y_h2 = [0] * self._nhidden2
            self._y_out = [0] * self._noutput

            #create weight from layers
            self._w_in_h1 = narray(self._ninput, self._nhidden1, 'r')
            self._w_h1_h2 = narray(self._nhidden1, self._nhidden2, 'r')
            self._w_h2_out = narray(self._nhidden2, self._noutput, 'r') 

            #create mommentum 
            self._mm_in_h1 = narray(self._ninput, self._nhidden1, 'r')
            self._mm_h1_h2 = narray(self._nhidden1, self._nhidden2, 'r')
            self._mm_h2_out = narray(self._nhidden2, self._noutput, 'r')

        else:
            print('error in arguments')

        self._input = None
        self._output = None
        self._lr = None
        self._it = None
        self._bias = None
        self._mm = None
        self._error_training = None
        self._mode_training = 'sequential'
        self._fx = None
        self._dfx = None


    def configuration(self, conf):
        ''' configuration of neural network with dict: conf '''

        self._input = conf['input']
        self._output = conf['output'] 
        self._lr = conf['lr']
        self._it = conf['it']
        self._bias = conf['bias']
        self._mm = conf['mm']
        self._error_training = conf['error']
        
        if 'mode' in conf.keys():
            self._mode_training = 'sequential'
        else:
            self._mode_training = conf.mode

        if conf['fx'] is None or conf['dfx'] is None:
            self._fx = ftanh
            self._dfx = dftanh
        else:
            self._fx = conf['fx']
            self._dfx = conf['dfx']  


    def training(self, p = True):
        ''' training neural network '''

        if self._mode_training == 'sequential':

            for it in range(self._it):

                error = 0.0

                for i,j in zip(self._input, self._output):
                    self.propagation(i)
                    error += self.backpropagation(j)

                if p is True:
                    print('iteration: ', it+1, 'error: ', error)

                if error <= self._error_training:
                    print("Learned Patterns!: N. of Iteractions: ",it+1)
                    print("Value error: ", error)
                    break

        elif self._mode_training == 'lot':
            print('training lot')
        else:
            print('not define mode training')




    def propagation(self, x):
        ''' Propagation of inputs x in layers '''

        self._y_in = x
        
        for j in range(self._nhidden1):
            sum_ = 0.0
            for i in range(self._ninput):
                sum_ += self._w_in_h1[i][j] * self._y_in[i]
            self._sum_h1[j] = sum_
            self._y_h1[j] = self._fx(sum_)

        if len(self._topology) == 4:

            for j in range(self._nhidden2):
                sum_ = 0.0
                for i in range(self._nhidden1):
                    sum_ += self._w_h1_h2[i][j] * self._y_h1[i]
                self._sum_h2[j] = sum_
                self._y_h2[j] = self._fx(sum_)

            for j in range(self._noutput):
                sum_ = 0.0
                for i in range(self._nhidden2):
                    sum_ += self._w_h2_out[i][j] * self._y_h2[i]
                self._sum_out[j] = sum_
                self._y_out[j] = self._fx(sum_)

        else:

            for j in range(self._noutput):
                sum_ = 0.0
                for i in range(self._nhidden1):
                    sum_ += self._w_h1_out[i][j] * self._y_h1[i]
                self._sum_out[j] = sum_
                self._y_out[j] = self._fx(sum_)


        return self._y_out


    def backpropagation(self, y):
        ''' Back-Propagation errors in layers '''

        #calcule error 
        lms_error = 0.0
        for i in range(len(y)):
            lms_error += (0.5 * ((y[i] - self._y_out[i]) ** 2))


        #calcule delta for layer output
        out_delta = [0] * self._noutput
        error = 0.0 
        for i in range(self._noutput):
            error = y[i] - self._y_out[i]
            out_delta[i] = error * self._dfx(self._sum_out[i])

        # calcule deltas in layers hiddens  
        if len(self._topology) == 4:

            h2_delta = [0] * self._nhidden2
            for i in range(self._nhidden2):
                error = 0.0
                for j in range(self._noutput):
                    error += out_delta[j] * self._w_h2_out[i][j]
                h2_delta[i] = self._dfx(self._sum_h2[i]) * error

            h1_delta = [0] * self._nhidden1
            for i in range(self._nhidden1):
                error = 0.0
                for j in range(self._nhidden2):
                    error += h2_delta[j] * self._w_h1_h2[i][j]
                h1_delta[i] = self._dfx(self._sum_h1[i]) * error

            #update weights and mommentum
            for i in range(self._nhidden2):
                mod = 0.0
                for j in range(self._noutput):
                    mod = out_delta[j] * self._y_h2[i]
                    self._w_h2_out[i][j] += (self._lr * mod) 
                    + (self._mm * self._mm_h2_out[i][j])
                    self._mm_h2_out[i][j] = mod

            for i in range(self._nhidden1):
                mod = 0.0
                for j in range(self._nhidden2):
                    mod = h2_delta[j] * self._y_h1[i]
                    self._w_h1_h2[i][j] += (self._lr * mod) 
                    + (self._mm * self._mm_h1_h2[i][j])
                    self._mm_h1_h2[i][j] = mod

            for i in range(self._ninput):
                mod = 0.0
                for j in range(self._nhidden1):
                    mod = h1_delta[j] * self._y_in[i]
                    self._w_in_h1[i][j] += (self._lr * mod) 
                    + (self._mm * self._mm_in_h1[i][j])
                    self._mm_in_h1[i][j] = mod

        else:
            # if hidden2 not exist
            h1_delta = [0] * self._nhidden1
            for i in range(self._nhidden1):
                error = 0.0
                for j in range(self._noutput):
                    error += out_delta[j] * self._w_h1_out[i][j]
                h1_delta[i] = self._dfx(self._sum_h1[i]) * error

            #update weights and mommentum
            for i in range(self._nhidden1):
                mod = 0.0
                for j in range(self._noutput):
                    mod = out_delta[j] * self._y_h1[i]
                    self._w_h1_out[i][j] += (self._lr * mod) 
                    + (self._mm * self._mm_h1_out[i][j])
                    self._mm_h1_out[i][j] = mod

            for i in range(self._ninput):
                mod = 0.0
                for j in range(self._nhidden1):
                    mod = h1_delta[j] * self._y_in[i]
                    self._w_in_h1[i][j] += (self._lr * mod) 
                    + (self._mm * self._mm_in_h1[i][j])
                    self._mm_in_h1[i][j] = mod

        return lms_error


    def test(self, x):

        for i in x:
            print(self.propagation(i)[0])

    # ----------- in construction -----------------
    def salve_state(self, name_file):
        pass

    def load_state(self, name_file):
        pass

    def load_input(self,name_file):
        pass

    def load_output(self,name_file):
        pass






