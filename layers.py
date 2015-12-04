import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def tanH(x):
    y = T.tanh(x)
    return(y)
    
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
    
    
def RelU(x, alpha ):
    #y = T.nnet.relu(x, alpha = alpha)
    y = T.switch(x > 0,x,alpha*x)
    return(y)        
    

class layer(object):
    def __init__(
        self,
        input,
        n_in,
        n_out,
        activation,
        numpy_rng,
        W,
        flagy = True
        ):
 
        if flagy is True:
         print ".... something in layer initially"
        
        if W is None:
            initial_W = numpy.asarray(
                    numpy_rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                        high=4 * numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                        ),
                    dtype=theano.config.floatX
                    )
            W = theano.shared(value=initial_W, name='Weights', borrow=True)
        
        b = theano.shared(
                value=numpy.zeros(
                        n_out,
                        dtype=theano.config.floatX
                        ), name = 'Bias',
                        borrow=True
                )
        
        self.output = activation (T.dot(input,W)+b) 
        self.params = [W, b]
      
    
    
    
    
