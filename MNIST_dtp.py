"""
Recurrent Neural Network with Target Propagation Through Time

(C) 2018 Nikolay Manchev
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

This code implements the TPTT-RNN network as described in the paper

Manchev, N. and Spratling, M., "Target Propagation in Recurrent Neural Networks", Journal of Machine Learning Research 21 (2020) 1-33

The underlying SRN implementation is based on code from 

Pascanu, R. and Mikolov, T. and Bengio, Y, "Understanding the exploding 
gradient problem.", https://arxiv.org/abs/1211.5063 

This network also uses the original data generation classes (TempOrderTask,
AddTask, PermTask, TempOrder3bitTask) used in Pascanu et. al., available at 
https://github.com/pascanur/trainingRNNs

The orthogonal initialisation code comes from
Lee, D. H. and Zhang, S. and Fischer, A. and Bengio, Y., Difference 
Target Propagation, CoRR, abs/1412.7525, 2014, https://github.com/donghyunlee/dtp
"""

import theano
import math
import time
import datetime
import numpy as np
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

from sklearn.preprocessing import OneHotEncoder  
from collections import OrderedDict

# Set default Theano attributes
theano.config.floatX="float32"
theano.config.optimizer="fast_run"

def vanilla_sgd(params, grads, learning_rate):
    """
    Update rules for vanilla SGD. Based on the update functions from
    Lasagne (https://github.com/Lasagne/Lasagne)
    
    The update is computed as
    
        param := param - learning_rate * gradient
        
    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression
    
    """
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates


def nesterov_momentum(params, grads, learning_rate, momentum=0.9):
    """
    Update rules for Nesterov accelerated gradient descent. Based on the update 
    functions from Lasagne (https://github.com/Lasagne/Lasagne)
    
    The update is computed as
    
        velocity[t] := momentum * velocity[t-1] + learning_rate * gradient[t-1]
        param       := param[t-1] + momentum * velocity[t] 
                                  - learning_rate * gradient[t-1]
        
    Parameters
    ----------
    params        : list of shared varaibles that will be updated
    grads         : list of symbolic expressions that produce the gradients
    learning_rate : step size
    momentum      : amount of momentum
    
    Returns
    -------
    A dictionary mapping each parameter in params to their update expression
    with applied momentum
    
    """
    updates = vanilla_sgd(params, grads, learning_rate)
    
    for param in params:
      value = param.get_value(borrow=True)
      velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
      x = momentum * velocity + updates[param] - param
      updates[velocity] = x
      updates[param] = momentum * x + updates[param]

    return updates

def load_MNIST(data_folder, one_hot = False, norm = True, sample_train = 0, sample_test = 0):
    """
    Loads, samples (if needed), and one-hot encodes the MNIST data set.
            
    Parameters
    ----------
    data_folder   : location of the MNIST data
    one_hot       : if True the target labels will be one-hot encoded
    norm          : if True the images will be normalised
    sample_train  : fraction of the train data to use. if set to 0 no sampling
                    will be applied (i.e. 100% of the data is used)
    sample_train  : fraction of the test data to use. if set to 0 no sampling
                    will be applied (i.e. 100% of the data is used)
    Returns
    -------
    X_train - Training images. Dimensions are (784, number of samples, 1)
    y_train - Training labels. Dimensions are (number of samples, 10)
    X_test  - Test images. Dimensions are (784, number of samples, 1)
    y_test  - Test labels. Dimensions are (number of samples, 10)
    """
    X_train = np.genfromtxt("%s/train_X.csv" % data_folder, delimiter=',')
    y_train = np.asarray(np.fromfile("%s/train_y.csv" % data_folder, sep='\n'), dtype="int32")

    X_test = np.genfromtxt("%s/test_X.csv" % data_folder, delimiter=',')
    y_test = np.asarray(np.fromfile("%s/test_y.csv" % data_folder, sep='\n'), dtype="int32")   

    if (sample_train != 0) and (sample_test != 0):
        
        print("Elements in train : %i" % sample_train)
        print("Elements in test  : %i" % sample_test)
        
        idx_train = np.random.choice(np.arange(len(X_train)), sample_train, replace=False)
        idx_test = np.random.choice(np.arange(len(X_test)), sample_test, replace=False)
        
        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

        
    if norm:
        print("MNIST NORMALISED!")
        X_train /= 255
        X_test  /= 255

    # Swap axes
    X_train = np.swapaxes(np.expand_dims(X_train,axis=0),0,2)
    X_test = np.swapaxes(np.expand_dims(X_test,axis=0),0,2)
    
    # Encode the target labels
    if one_hot:
        onehot_encoder = OneHotEncoder(sparse=False, categories="auto")        
        y_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))
        y_test = onehot_encoder.fit_transform(y_test.reshape(-1,1))
        
    return X_train, y_train, X_test, y_test

def mse(x,y):
    """
    Computes the mean squared error. The average is performed element-wise 
    along the squared difference, returning a single value.
    
    Parameters
    ----------
    x, y : symbolic tensors
    
    Returns
    -------
    MSE(x,y)
    
    """
    return T.mean((x-y) ** 2) 

def gaussian(shape, std):
    """
    Draw random samples from a normal distribution.
    
    Parameters
    ----------
    shape      : output shape
    std        : standard deviation
    
    Returns
    -------
    Drawn samples from the parameterized normal distribution
    """    
    rng = RandomStreams(seed=1234)
    return rng.normal(std=std, size=shape)

def rand_ortho(shape, irange, rng):
    """
    Generates an orthogonal matrix. Original code from 
    
    Lee, D. H. and Zhang, S. and Fischer, A. and Bengio, Y., Difference 
    Target Propagation, CoRR, abs/1412.7525, 2014
    
    https://github.com/donghyunlee/dtp
    
    Parameters
    ----------
    shape  : matrix shape
    irange : range for the matrix elements
    rng    : RandomState instance, initiated with a seed
     
    Returns
    -------
    An orthogonal matrix of size *shape*        
    """
    A = - irange + 2 * irange * rng.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return np.asarray(np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V )),
                      dtype = theano.config.floatX)  

def fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, 
        batch_size, maxepoch, chk_interval, gaussian_noise, gd_opt,
        x_train_data, y_train_data, x_test_data, y_test_data, threshold=100):
    """
    Fits a TPTT-trained SRN model on MNIST.
    
    Parameters
    ----------
    rng             : RandomState instance, initiated with a seed
    i_learning_rate : initial learning rate (alpha_i)
    f_learning_rate : forward learning rate (alpha_f)
    g_learning_rate : feedback learning rate (alpha_g)
    n_hid           : number of neurons in the hidden layer
    init            : hidden units initialisation
    batch_size      : number of samples per mini-batch
    maxepoch        : maximal number of training epochs
    chk_interval    : number of samples between validation steps
    gaussian_noise  : amount of injected Gaussian noise
    gd_opt          : optimisation technique (vanilla sgd, nesterov)
    x_train_data    : MNIST training images
    y_train_data    : MNIST training labels
    x_test_data     : MNIST test images
    y_test_data     : MNIST test labels
    threshold       : paitence threshold -- if no improvement is seen over
                      this number of validation checks the training is aborted
    
    """
    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - Simple RNN TPTT on MNIST")
    print("******************************************************")
    print("optimization: %s" % gd_opt)
    print("i_learning_rate: %.10f" % i_learning_rate)
    print("f_learning_rate: %.10f" % f_learning_rate)
    print("g_learning_rate: %.10f" % g_learning_rate)
    print("noise: %f" % gaussian_noise)
    print("maxepoch: %i" % maxepoch)
    print("batch_size: %i" % batch_size)
    print("chk_interval: %i" % chk_interval)
    print("n_hid: %i" % n_hid)
    print("init: %s" % init)
    print("wxh_updates: tptt")
    print("paitence threshold: %i" % threshold)
    print("******************************************************")
    
    X_train = theano.shared(np.asarray(x_train_data,dtype=theano.config.floatX), 
                            name = "X_train", borrow=True)
                         
    y_train = theano.shared(np.asarray(y_train_data, dtype="int32"), 
                            name = "y_train", borrow=True)

    n_inp = 1
    n_out = 10
    
    if init == "sigmoid":
        Wxh = np.asarray(rng.normal(size=(n_inp, n_hid), scale=.01, loc =.0), dtype = theano.config.floatX)
        Whh = np.asarray(rng.normal(size=(n_hid, n_hid), scale=.01, loc =.0), dtype = theano.config.floatX)
        Why = np.asarray(rng.normal(size=(n_hid, n_out), scale=.01, loc =.0), dtype = theano.config.floatX)
        bh  = np.zeros((n_hid,), dtype=theano.config.floatX)
        by  = np.zeros((n_out,), dtype=theano.config.floatX)
        
        activ = T.nnet.sigmoid

        Vhh = np.asarray(rng.normal(size=(n_hid, n_hid), scale=.01, loc = .0), dtype = theano.config.floatX)
        ch  = np.zeros((n_hid,), dtype=theano.config.floatX)

    elif init == "tanh-randorth":
        
        Whh = rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid +n_hid)), rng)
        bh  = np.zeros((n_hid,), dtype=theano.config.floatX)
        by  = np.zeros((n_out,), dtype=theano.config.floatX)
                
        Wxh = rand_ortho((n_inp, n_hid), np.sqrt(6./(n_inp +n_hid)), rng)
        Why = rand_ortho((n_hid, n_out), np.sqrt(6./(n_hid +n_out)), rng)
        
        activ = T.tanh

        Vhh = rand_ortho((n_hid, n_hid),np.sqrt(6./(n_hid +n_hid)), rng)
        ch  = np.zeros((n_hid,), dtype=theano.config.floatX)

    Wxh = theano.shared(Wxh, "Wxh")
    Whh = theano.shared(Whh, "Whh")
    Why = theano.shared(Why, "Why")
    bh  = theano.shared(bh, "bh")
    by  = theano.shared(by, "by")

    Vhh = theano.shared(Vhh, "Vhh")
    ch  = theano.shared(ch, "ch")
    
    #########################################
    # TRAINING
    #########################################

    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), batch_size, n_hid)

    x = T.tensor3()
    t = T.imatrix()

    i_lr = T.scalar()
    f_lr = T.scalar()
    g_lr = T.scalar()

    noise = T.scalar()
    
    F = lambda x, hs: activ(T.dot(hs, Whh) + T.dot(x, Wxh) + bh)
    
    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why, bh: F(x_t, h_prev), 
                       sequences = x,
                       outputs_info = [h0], # initialisation
                       non_sequences = [Whh, Wxh, Why, bh],
                       strict = True,
                       name = 'rec_layer')
                       #,mode = theano.Mode(linker='cvm'))

    # lastSoftmax    
    y = T.nnet.softmax(T.dot(h[-1], Why) + by)
    cost = -(t * T.log(y)).mean(axis=0).sum()

    # Setup targets    
    G = lambda x, hs: activ(T.dot(x, Wxh) + T.dot(hs, Vhh) + ch)

    first_target = h[-1] - i_lr * T.grad(cost, h)[-1]

    """
    h_ contains the local targets
    
    first_target - deepest hidden layer (e.g. H10)
    h_[:,0,:][0] - second deepest layer (e.g H9)]
    ...
    h_[:,0,:][len(h_)-1] - first hidden layer (e.g. H1)
    
    """
    
    h_,_ = theano.scan(fn = lambda x_tp1, h_t, h_tp1, h_hat_tp1, Wxh, Vhh, ch: h_t - G(x_tp1, h_tp1) + G(x_tp1, h_hat_tp1),
                     sequences = [dict(input=x, taps=[0]),dict(input=h, taps=[-1]), dict(input=h, taps=[0])],
                     outputs_info = [first_target], # initialisation
                     non_sequences = [Wxh, Vhh, ch],
                     go_backwards = True,
                     strict = True,
                     name = "target_layer")
        
    # Merge first_target & h_ and get an unified tensor with all targets
    first_target = T.reshape(first_target, [1,first_target.shape[0],first_target.shape[1]])
    h_ = T.concatenate([first_target, h_])

    # Reverse the order of h_ to get [H_0 H_1 H_2 ....]
    h_ = h_[::-1]
    
    # gradients of feedback (inverse) mapping
            
    # Splice h0 and h in h_offset, and remove H for the last layer (we don't need it)    
    h_offset = T.concatenate([T.reshape(h0, [1,h0.shape[0],h0.shape[1]]), h])[:-1,:,:]

    # Add gaussian noise
    h_offset_c = h_offset + gaussian(h_offset.shape, noise)
    
    # Loop over h_offset & x so that T.grad(mse(G(x[t],F(x[t], h[t-1])),h[t-1]), [Vhh,Ch], consider_constant=[x[t],F(x[t], h[t-1]),h[t-1]])    
    (dVhh, dCh), _ = theano.scan(fn = lambda x_t, h_tm1, Wxh, Vhh, ch: T.grad(mse(G(x_t,F(x_t, h_tm1)),h_tm1), [Vhh,ch], consider_constant=[G(x_t,F(x_t, h_tm1)),h_tm1]),
                          sequences = [x, h_offset_c],
                          non_sequences = [Wxh, Vhh, ch],
                          name = "feedback_mapping")
    
    # This is probably incorrect dCh = T.grad(mse(G(x,F(x, h_offset)),h_offset), ch, consider_constant=[x,F(x, h_offset),h_offset])
    
    # add up all corrections
    dVhh = dVhh.sum(axis=0)
    dVhh.name="dVhh"
    
    dCh  = dCh.sum(axis=0)
    dCh.name="dCh"
    
    g_norm_theta = T.sqrt((dVhh**2).sum() + (dCh**2).sum())

    # gradients of feedforward
    (dWhh,dbh,dWxh),_ = theano.scan(fn = lambda x_t, h_tm1, h_hat_t, Wxh, Whh, bh: T.grad(mse(F(x_t, h_tm1),h_hat_t), [Whh,bh,Wxh], consider_constant=[F(x_t, h_tm1),h_hat_t]),
                          sequences = [x, h_offset, h_],
                          non_sequences = [Wxh, Whh, bh],
                          strict = True,
                          name = "feedforward_mapping")

    dWhh = dWhh.sum(axis=0)
    dbh  = dbh.sum(axis=0)
    dWxh = dWxh.sum(axis=0)
    
    dWhy, dby = T.grad(cost, [Why, by])

    # Set the optimisation technique
    if gd_opt == "vanilla":
        # Vanilla SGD
        updates_g = vanilla_sgd([Vhh, ch], [dVhh,dCh], g_lr)
        updates_f = vanilla_sgd([Wxh, Whh, bh, Why, by],[dWxh, dWhh, dbh, dWhy, dby], f_lr )            

    elif gd_opt == "nesterov":
        # Nesterov accelerated gradient
        updates_g = nesterov_momentum([Vhh, ch], [dVhh, dCh], g_lr)
        updates_f = nesterov_momentum([Wxh, Whh, bh, Why, by],[dWxh, dWhh, dbh, dWhy, dby], f_lr)        

    dWhh_norm = T.sqrt((dWhh**2).sum())
    dWxh_norm = T.sqrt((dWxh**2).sum())
    dWhy_norm = T.sqrt((dWhy**2).sum())
    dby_norm = T.sqrt((dby**2).sum())
    dbh_norm = T.sqrt((dbh**2).sum())
    
    minibatch_index = T.lscalar("minibatch_index")

    givens_f_step = {
        x : X_train[:,minibatch_index * batch_size: (minibatch_index + 1) * batch_size,:],
        t : y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size,:]
    }

    f_step = function([minibatch_index,i_lr, f_lr],
                      [cost, dWhh_norm, dWxh_norm, dWhy_norm, dby_norm, dbh_norm],
                      on_unused_input='warn',
                      updates = updates_f,
                      givens = givens_f_step)

    
    g_step = function([minibatch_index,g_lr,noise], 
                      g_norm_theta,
                      on_unused_input='warn',
                      updates = updates_g,
                      givens = {x : X_train[:,minibatch_index * batch_size: (minibatch_index + 1) * batch_size,:]})
    
    #########################################
    # VALIDATION
    #########################################
    X_test = theano.shared(np.asarray(x_test_data,dtype=theano.config.floatX), 
                           name = "X_test", borrow=True)
    
    y_test = theano.shared(np.asarray(y_test_data, dtype="int32"), 
                            name = "y_test", borrow=True)

    x = T.tensor3()
    t = T.imatrix()

    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), X_test.get_value().shape[1], n_hid)

    givens_val = {x: X_test, 
                  t: y_test}

    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why: activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x,
                       outputs_info = [h0],
                       non_sequences = [Whh, Wxh, Why],
                       name = 'validation')

    # lastSoftmax    
    y = T.nnet.softmax(T.dot(h[-1], Why) + by)
    cost = -(t * T.log(y)).mean(axis=0).sum()
    error = T.neq(T.argmax(y, axis=1), T.argmax(t, axis=1)).mean()        
    
    eval_step = function([], 
                         [cost, error],
                         givens = givens_val
                         )
    
    print("******************************************************")
    print("Training starts...")
    print("******************************************************")

    training = True
    
    n = 0

    acc = []    
    
    n_minibatches = math.ceil(X_train.get_value().shape[1] / batch_size)
    
    paitence = 0
    
    best_acc = 0

    total_samples = batch_size*n_minibatches*maxepoch
    zero_padding = "%0" + str(len(str(total_samples))) + "d"    
    
    start_time = time.time()  
         
    print(str(datetime.datetime.now()).split('.', 2)[0])    

    while (training) and (n < maxepoch):

        n += 1
        
        avg_cost = 0
        avg_dWhh_norm = 0
        avg_dWxh_norm = 0
        avg_dWhy_norm = 0
        avg_dbh_norm = 0
        avg_dhy_norm = 0
        
        avg_g_norm = 0               
        
        for minibatch_index in range(n_minibatches):            
            
            g_norm = g_step(minibatch_index, g_learning_rate, gaussian_noise)

            tr_cost, f_Whh, f_Wxh, f_Why, f_by, f_bh = f_step(minibatch_index, i_learning_rate, f_learning_rate)

            avg_cost += tr_cost
    
            avg_dWhh_norm += f_Whh
            avg_dWxh_norm += f_Wxh
            avg_dWhy_norm += f_Why
            avg_dbh_norm += f_bh
            avg_dhy_norm += f_by
    
            avg_g_norm += g_norm

            samples_seen = ((n - 1) * n_minibatches + (minibatch_index+1)) * batch_size

            if (samples_seen % chk_interval == 0):

                elapsed_time = time.time() - start_time                
                
                if minibatch_index != 0:
                    avg_cost = avg_cost / float(chk_interval)
                    avg_g_norm = avg_g_norm / float(chk_interval)                    
        
                    avg_dWhh_norm = avg_dWhh_norm/ float(chk_interval)                    
                    avg_dWxh_norm = avg_dWxh_norm/ float(chk_interval)                    
                    avg_dWhy_norm = avg_dWhy_norm/ float(chk_interval)                    
                    avg_dbh_norm = avg_dbh_norm/ float(chk_interval)                    
                    avg_dhy_norm = avg_dhy_norm/ float(chk_interval)                    

                    remaining_time = (elapsed_time / samples_seen) * (total_samples - samples_seen)             
    
                valid_cost, error = eval_step()                
    
                acc.append((1.0 - error)*100)
                   
                if acc[-1] > best_acc:
                    best_acc = acc[-1]
                    paitence = 0
                else:
                    paitence += 1
                
                rho_Whh =np.max(abs(np.linalg.eigvals(Whh.get_value())))
                                        
                print("Epoch: %i, \t Samples seen: " % n,
                       zero_padding % samples_seen + "/%d, " % total_samples,
                      "cost %05.3f, " % avg_cost,
                      "|dWhh| %7.5f, " % avg_dWhh_norm,
                      "rho %01.3f, " % rho_Whh,
                      "|dbh| %7.3f, " % avg_dbh_norm,
                      "|dWxh| %7.3f, " % avg_dWxh_norm,
                      "|dWhy| %7.3f, " % avg_dWhy_norm,
                      "|dby| %7.3f, " % avg_dhy_norm,
                      "|dG| %7.5f, " % avg_g_norm,
                      "val err %05.2f%%, " % (error*100),
                      "best accuracy %05.2f%%, " % best_acc,
                      "paitence %i/%i" % (paitence, threshold))
    
        if paitence >= threshold:
            print("No improvement in %i iterations. Aborting...." % threshold)
            training = False
       
    print("******************************************************")
    print(str(datetime.datetime.now()).split('.', 2)[0])
    print("Training completed. Final best error : %07.3f%%" % best_acc)
    print("******************************************************")
    print("------------------------------------------------------")

    return (n-1), best_acc, smoothness

if __name__=='__main__':
    
    rng = np.random.RandomState(1234)
    
    sample_train = 10000
    sample_test  = 1000
    
    try:
      X_train_data
    except NameError:
      X_train_data, y_train_data, X_test_data, y_test_data = load_MNIST("mnist", one_hot=True, 
                                                                        norm=True, sample_train=sample_train, 
                                                                        sample_test=sample_test)
          
    maxepoch = 100
    batch_size = 16
    chk_interval = 2000
    n_hid = 100

    init   = "tanh-randorth"
    gd_opt = "vanilla"
        
    i_learning_rate = 0.0000001
    f_learning_rate = 0.01
    g_learning_rate = 0.00000001
    
    noise = 0.1
            
    fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init,
        batch_size, maxepoch, chk_interval, noise, gd_opt, X_train_data, 
        y_train_data, X_test_data, y_test_data)

