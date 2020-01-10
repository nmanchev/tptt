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

import numpy as np
import theano
import theano.tensor as T
import argparse
import sys

from theano import function

from tempOrder import TempOrderTask
from addition import AddTask
from permutation import PermTask
from tempOrder3bit import TempOrder3bitTask

from collections import OrderedDict

from theano.tensor.shared_randomstreams import RandomStreams

# Set default Theano attributes
theano.config.floatX="float32"
theano.config.optimizer="fast_compile"

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

def sample_length(min_length, max_length, rng):
    """
    Computes a sequence length based on the minimal and maximal sequence size.
    
    Parameters
    ----------
    max_length      : maximal sequence length (t_max)
    min_length      : minimal sequence length
    
    Returns
    -------
    A random number from the max/min interval
    """
    length = min_length
    
    if max_length > min_length:
        length = min_length + rng.randint(max_length - min_length)

    return length

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

def fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, 
        batch_size, max_length, min_length, task, maxiter, chk_interval,
        gaussian_noise, val_size, val_batch, gd_opt, task_name, wxh_updates):
    """
    Fits a TPTT-trained SRN model.
    
    Parameters
    ----------
    rng             : RandomState instance, initiated with a seed
    i_learning_rate : initial learning rate (alpha_i)
    f_learning_rate : forward learning rate (alpha_f)
    g_learning_rate : feedback learning rate (alpha_g)
    n_hid           : number of neurons in the hidden layer
    init            : hidden units initialisation
    batch_size      : number of samples per mini-batch
    max_length      : maximal sequence length (t_max)
    min_length      : minimal sequence length 
    task            : task type (addition, temporal order etc.)
    maxiter         : maximal number of iterations
    chk_interval    : number of iterations between validation
    gaussian_noise  : amount of injected Gaussian noise
    val_size        : size of the validation set
    val_batch       : number of samples in the validation set
    gd_opt          : optimisation technique (vanilla sgd, nesterov)
    task_name       : name of the synthetic problem
    wxh_updates     : update mechanism for Wxh
    """

    print("------------------------------------------------------")
    print("******************************************************")
    print("Parameters - Simple RNN TPTT")
    print("******************************************************")
    print("task: %s" % task_name)
    print("optimization: %s" % gd_opt)
    print("i_learning_rate: %f" % i_learning_rate)
    print("f_learning_rate: %f" % f_learning_rate)
    print("g_learning_rate: %f" % g_learning_rate)
    print("maxiter: %i" % maxiter)
    print("batch_size: %i" % batch_size)
    print("min_length: %i" % min_length)
    print("max_length: %i" % max_length)
    print("chk_interval: %i" % chk_interval)
    print("n_hid: %i" % n_hid)
    print("init: %s" % init)
    print("val_size: %i" % val_size)
    print("val_batch: %i" % val_batch)
    print("noise: %f" % gaussian_noise)    
    print("wxh_updates: %s" % wxh_updates)
    print("******************************************************")
    
    # Get the number of inputs and outputs from the task
    n_inp = task.nin
    n_out = task.nout

    # Initialise the model parameters at random based on the specified
    # activation function 
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
        Wxh = rand_ortho((n_inp, n_hid), np.sqrt(6./(n_inp +n_hid)), rng)
        Whh = rand_ortho((n_hid, n_hid), np.sqrt(6./(n_hid +n_hid)), rng)
        Why = rand_ortho((n_hid, n_out), np.sqrt(6./(n_hid +n_out)), rng)
        bh  = np.zeros((n_hid,), dtype=theano.config.floatX)
        by  = np.zeros((n_out,), dtype=theano.config.floatX)
        
        activ = T.tanh

        Vhh = rand_ortho((n_hid, n_hid),np.sqrt(6./(n_hid +n_hid)), rng)
        ch  = np.zeros((n_hid,), dtype=theano.config.floatX)

    # Store the parameters in shared variables
    Wxh = theano.shared(Wxh, 'Wxh')
    Whh = theano.shared(Whh, 'Whh')
    Why = theano.shared(Why, 'Why')
    bh  = theano.shared(bh, 'bh')
    by  = theano.shared(by, 'by')

    Vhh = theano.shared(Vhh, 'Vhh')
    ch  = theano.shared(ch, 'ch')
    
    #########################################
    # TRAINING PHASE                        #
    #########################################

    # The initial state h0 is initialised with 0's
    h0 = T.alloc(np.array(0, dtype=theano.config.floatX), batch_size, n_hid)

    # Define symbolic variables
    x = T.tensor3()
    t = T.matrix()

    i_lr = T.scalar()
    f_lr = T.scalar()
    g_lr = T.scalar()
    
    noise = T.scalar()    
        
    # Define the forward function F(.)
    F = lambda x, hs: activ(T.dot(hs, Whh) + T.dot(x, Wxh) + bh)
    
    # Compute the forward outputs
    h, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why, bh: F(x_t, h_prev), 
                       sequences = x,
                       outputs_info = [h0], # initialisation
                       non_sequences = [Whh, Wxh, Why, bh],
                       strict = True,
                       name = 'rec_layer')

    # Compute the final output based on the problem type (classifiaction or 
    # real-valued) and get the global loss
    if task.classifType == 'lastSoftmax':
        # Classification problem - set the last layer to softmax and use
        # cross-entropy loss
        y = T.nnet.softmax(T.dot(h[-1], Why) + by)
        cost = -(t * T.log(y)).mean(axis=0).sum()
    elif task.classifType == 'lastLinear':
        # Real valued output - final step is linear, and the loss is MSE        
        y = T.dot(h[-1], Why) + by
        cost = ((t - y)**2).mean(axis=0).sum()

    # Define the G(.) function
    G = lambda x, hs: activ(T.dot(x, Wxh) + T.dot(hs, Vhh) + ch)

    # First target is based on the derivative of the global error w.r.t.
    # the parameters in the final layer
    first_target = h[-1] - i_lr * T.grad(cost, h)[-1]

    # Set the local targets for the upstream layers    
    # h_ contains the local targets
    # first_target - deepest hidden layer (e.g. H10)
    # h_[:,0,:][0] - second deepest layer (e.g H9)]
    # ...
    # h_[:,0,:][len(h_)-1] - first hidden layer (e.g. H1)
    
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
    
    # Add up all corrections
    dVhh = dVhh.sum(axis=0)    
    dCh  = dCh.sum(axis=0)
    
    # Compute the norm of the updates of G(.)    
    g_norm_theta = T.sqrt((dVhh**2).sum() + (dCh**2).sum())        
    
    # Gradients of the feedforward    
    if (wxh_updates == "bptt"):
        # Using TPTT for the Whh and bh updates        
        (dWhh,dbh),_ = theano.scan(fn = lambda x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh: T.grad(mse(F(x_t, h_tm1),h_hat_t), [Whh,bh], consider_constant=[x_t, h_tm1, h_hat_t]),
                              sequences = [x, h, h_offset, h_],
                              non_sequences = [Wxh, Whh, bh],
                              strict = True,
                              name = "feedforward_mapping")        
        # Using BPTT for the Wxh, Why, and by updates
        dWxh, dWhy, dby = T.grad(cost, [Wxh, Why, by])
    else:
        # Using TPTT for the Whh, bh, and Wxh updates
        (dWhh,dbh, dWxh),_ = theano.scan(fn = lambda x_t, h_t, h_tm1, h_hat_t, Wxh, Whh, bh: T.grad(mse(F(x_t, h_tm1),h_hat_t), [Whh,bh,Wxh], consider_constant=[x_t, h_tm1, h_hat_t]),
                              sequences = [x, h, h_offset, h_],
                              non_sequences = [Wxh, Whh, bh],
                              strict = True,
                              name = "feedforward_mapping")
        # Add up the corrections for Wxh
        dWxh = dWxh.sum(axis=0)
        # Compute dWhy and dby using BPTT
        dWhy, dby = T.grad(cost, [Why, by])

    # Add up the dWhh and dbh corrections
    dWhh = dWhh.sum(axis=0)
    dbh  = dbh.sum(axis=0)

    # Set the optimisation technique
    if gd_opt == "vanilla":
        # Vanilla SGD
        updates_g = vanilla_sgd([Vhh, ch], [dVhh,dCh], g_lr)
        updates_f = vanilla_sgd([Wxh, Whh, bh, Why, by],[dWxh, dWhh, dbh, dWhy, dby], f_lr )            

    elif gd_opt == "nesterov":
        # Nesterov accelerated gradient
        updates_g = nesterov_momentum([Vhh, ch], [dVhh, dCh], g_lr)
        updates_f = nesterov_momentum([Wxh, Whh, bh, Why, by],[dWxh, dWhh, dbh, dWhy, dby], f_lr)
    
    # Compute the norm of each feedforward update matrix
    dWhh_norm = T.sqrt((dWhh**2).sum())
    dWxh_norm = T.sqrt((dWxh**2).sum())
    dWhy_norm = T.sqrt((dWhy**2).sum())
    dby_norm = T.sqrt((dby**2).sum())
    dbh_norm = T.sqrt((dbh**2).sum())

    # Define a forward step function
    f_step = function([x,t,i_lr, f_lr],
                      [cost, dWhh_norm, dWxh_norm, dWhy_norm, dby_norm, dbh_norm],
                      on_unused_input='warn',
                      updates = updates_f)
    
    # Define a feedback step function
    g_step = function([x,g_lr,noise], 
                      g_norm_theta,
                      on_unused_input='warn',
                      updates = updates_g)
    
    #########################################
    # VALIDATION PHASE                      #
    #########################################
    
    # Define symbolic variables for the validation phase
    h0_val = T.alloc(np.array(0, dtype=theano.config.floatX), val_batch, n_hid)

    x_val = T.tensor3()
    t_val = T.matrix()

    # Set a forward pass
    h_val, _ = theano.scan(fn = lambda x_t, h_prev, Whh, Wxh, Why: activ(T.dot(h_prev, Whh) + T.dot(x_t, Wxh) + bh), 
                       sequences = x_val,
                       outputs_info = [h0_val],
                       non_sequences = [Whh, Wxh, Why],
                       name = "validation")

    # Compute the final output based on the problem type (classifiaction or 
    # real-valued), get the global loss, and measure the prediction error
    if task.classifType == "lastSoftmax":
        # Classification problem - set the last layer to softmax and use
        # cross-entropy loss
        y_val = T.nnet.softmax(T.dot(h_val[-1], Why) + by)
        cost_val = -(t_val * T.log(y_val)).mean(axis=0).sum()
        error_val = T.neq(T.argmax(y_val, axis=1), T.argmax(t_val, axis=1)).mean()        
    elif task.classifType == "lastLinear":
        # Real valued output - final step is linear, and the loss is MSE
        y_val = T.dot(h_val[-1], Why) + by
        cost_val = ((t_val - y_val)**2).mean(axis=0).sum()
        # An example in the mini-batch is considered successfully predicted if 
        # the error between the prediction and the target is below 0.04
        error_val = (((t_val - y_val)**2).sum(axis=1) > .04).mean()
        
    # Define a step function for the validation pass
    eval_step = function([x_val,t_val], [cost_val, error_val])

    print("******************************************************")
    print("Training starts...")
    print("******************************************************")
    
    # Control variable for the tarining loop
    training = True
    
    # Iteration number
    n = 1
    
    # Cost accumulator variable
    avg_cost = 0
    
    # Gradient norm accumulator variables
    avg_dWhh_norm = 0
    avg_dWxh_norm = 0
    avg_dWhy_norm = 0
    avg_dbh_norm = 0
    avg_dhy_norm = 0
    avg_g_norm = 0
    
    patience = 300
    
    # Measure the initial accuracy
    valid_x, valid_y = task.generate(val_batch, 
                                     sample_length(min_length, 
                                                   max_length, rng))
    best_score = eval_step(valid_x, valid_y) [1] * 100                   
    
    # Repeat until convergence or upon reaching the maxiter limit
    while (training) and (n <= maxiter):
            
         # Get a mini-batch of training data
        train_x, train_y = task.generate(batch_size, 
                                         sample_length(min_length, 
                                                       max_length, rng))
        
        # Perform a feedback step (set targets)
        g_norm = g_step(train_x, g_learning_rate, gaussian_noise)

        # Perform a forward step
        tr_cost, f_Whh, f_Wxh, f_Why, f_by, f_bh = f_step(train_x, train_y, i_learning_rate, f_learning_rate)

        # Update the accumulation variables
        avg_cost += tr_cost

        avg_dWhh_norm += f_Whh
        avg_dWxh_norm += f_Wxh
        avg_dWhy_norm += f_Why
        avg_dbh_norm += f_bh
        avg_dhy_norm += f_by
        avg_g_norm += g_norm
        
        if (n % chk_interval == 0):
            patience = patience - 1
            # Time to check the performance on the validation set
            
            # If the cost is NAN, abort the training
            
            avg_cost = avg_cost / float(chk_interval)
            
            if not np.isfinite(tr_cost):
                print("******************************************************")
                print("Cost is NAN. Training aborted. Best error : %07.3f%%" % best_score)
                print("******************************************************")
                print("------------------------------------------------------")
                return (n-1),best_score
            
            # Get the average of the accumulation variables
            
            avg_g_norm = avg_g_norm / float(chk_interval)                    

            avg_dWhh_norm = avg_dWhh_norm/ float(chk_interval)                    
            avg_dWxh_norm = avg_dWxh_norm/ float(chk_interval)                    
            avg_dWhy_norm = avg_dWhy_norm/ float(chk_interval)                    
            avg_dbh_norm = avg_dbh_norm/ float(chk_interval)                    
            avg_dhy_norm = avg_dhy_norm/ float(chk_interval)                    

            # Accumulation variables for the validation cost and error
            valid_cost = 0
            error = 0

            # Get the number of mini-batches needed to cover the desired
            # validation sample and loop over them            
            for dx in range(val_size // val_batch):
                    
                # Get a mini-batch for validation
                valid_x, valid_y = task.generate(val_batch, 
                                                 sample_length(min_length, 
                                                               max_length, rng))

                # Take a validation step and get the cost and error from
                # this mini-batch
                _cost, _error = eval_step(valid_x, valid_y)                
                error = error + _error
                valid_cost = valid_cost + _cost
 
            # Compute the average error and cost
            error = error*100. / float(val_size // val_batch)
            valid_cost = valid_cost / float(val_size // val_batch)

            # Get the spectral radius of the Whh and Vhh matrices            
            rho_Whh =np.max(abs(np.linalg.eigvals(Whh.get_value())))
            rho_Vhh =np.max(abs(np.linalg.eigvals(Vhh.get_value())))

            if (rho_Whh>20 or rho_Vhh>20):
                print("Rho exploding. Aborting....")
                training = False

            # Is the new error lower than our best? Update the best
            if error < best_score:
                patience = 300
                best_score = error
                    
            if (patience <= 0):
                print("No improvement over 30'000 samples. Aborting...")
                training = False
                
            # Print the results from the validation
            print("Iter %07d" % n, ":",
                  "cost %05.3f, " % avg_cost,
                  "|Whh| %7.5f, " % avg_dWhh_norm,
                  "r %01.3f," % rho_Whh,
                  "|bh| %7.3f, " % avg_dbh_norm,
                  "|Wxh| %7.3f, " % avg_dWxh_norm,
                  "|Why| %7.3f, " % avg_dWhy_norm,
                  "|by| %7.3f, " % avg_dhy_norm,
                  "|g| %7.5f, " % avg_g_norm,
                  "r %01.3f," % rho_Vhh,
                  "err %07.3f%%, " % error,
                  "best err %07.3f%%" % best_score)

            # Is the error below 0.0001? If yes, the problem has been solved
            if error < .0001 and np.isfinite(valid_cost):
                training = False
                print("PROBLEM SOLVED!")


            # Reset the accumulators
            avg_cost = 0
            avg_dWhh_norm = 0
            avg_dWxh_norm = 0
            avg_dWhy_norm = 0
            avg_dbh_norm = 0
            avg_dhy_norm = 0
            
            avg_g_norm = 0

        # Increase the iteration counter
        n += 1    
        
    # Training completed. Print the final validation error.
    print("******************************************************")
    print("Training completed. Final best error : %07.3f%%" % best_score)
    print("******************************************************")
    print("------------------------------------------------------")

    return (n-1),best_score


def main(args): 

    # Set a random seed for reproducibility
    rng = np.random.RandomState(1234)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Runs a TPTT RNN test against the pathological\
                                     tasks defined in Hochreiter, S. and Schmidhuber, J. \
                                     (1997). Long short-term memory. Neural Computation, \
                                     9(8), 1735–1780\nThis work is licensed under the \
                                     Creative Commons Attribution 4.0 International License.")

    parser.add_argument("--task", help="Pathological task", choices=["temporal", "temporal3", "addition", "perm"],
                        required=True)

    parser.add_argument("--maxiter",help="Maximum number of iterations", 
                        default = 100000, required=False, type=int)
    
    parser.add_argument("--batchsize",help="Size of the minibatch", 
                        default = 20, required=False, type=int)

    parser.add_argument("--min",help="Minimal length of the task", 
                        default = 10, required=False, type=int)

    parser.add_argument("--max",help="Maximal length of the task", 
                        default = 10, required=False, type=int)

    parser.add_argument("--chk",help="Check interval", 
                        default = 100, required=False, type=int)

    parser.add_argument("--hidden",help="Number of units in the hidden layer", 
                        default = 100, required=False, type=int)

    parser.add_argument("--opt", help="Optimizer", choices=["vanilla", "nesterov"],
                        default = "nesterov", required=False)

    parser.add_argument("--init", help="Weight initialization and activation function", choices=["tanh-randorth", "sigmoid"],
                        default = "tanh-randorth", required=False)

    parser.add_argument("--ilr",help="Initial learning rate", default = 0.1, 
                        required=False, type=float)

    parser.add_argument("--flr",help="Forward learning rate", default = 0.01, 
                        required=False, type=float)

    parser.add_argument("--glr",help="Feedback learning rate", default = 0.001, 
                        required=False, type=float)
    
    parser.add_argument("--wxh_updates", help="Update mechanism for Wxh", choices=["bptt", "tptt"],
                        default = "tptt", required=False)

    parser.add_argument("--noise", help="Injected Gaussian noise", 
                        default = 0.0, required=False, type=float)

    args = parser.parse_args()

    # Maximal length of the task and minimal length of the task.
    # If you want to run an experiment were sequences have fixed length, set
    # these to hyper-parameters to the same value. Otherwise each batch will
    # have a length randomly sampled from [min_length, max_length]
    min_length = args.min
    max_length = args.max
    
    noise = args.noise
    
    # Get the problem type and instantiate the respective generator
    if args.task == "temporal":
        task = TempOrderTask(rng, theano.config.floatX)        
    if args.task == "temporal3":
        task = TempOrder3bitTask(rng, theano.config.floatX)        
    elif args.task == "addition":
        task = AddTask(rng, theano.config.floatX)
    elif args.task == "perm":
        task = PermTask(rng, theano.config.floatX)
    
    # Set the maximum number of iterations
    maxiter = args.maxiter
    
    # Update mechanism for Wxh
    wxh_updates = args.wxh_updates

    # Set the mini-batch size
    batch_size = args.batchsize

    # Set the number of iterations between each validation
    chk_interval = args.chk

    # Set the number of neurons in the hidden layer
    n_hid = args.hidden

    # Set the random weights initialisation and the optimisation techniuqe
    init   = args.init
    gd_opt = args.opt
    
    # Set the size and number of mini-batches for the validation phase
    val_size  = 10000
    val_batch = 1000
    
    # Set the learning rates and Gaussian noise decay iteration
    i_learning_rate = args.ilr
    f_learning_rate = args.flr
    g_learning_rate = args.glr
    
    # Train the network
    fit(rng, i_learning_rate, f_learning_rate, g_learning_rate, n_hid, init, \
        batch_size, max_length, min_length, task, maxiter, chk_interval, noise, \
        val_size, val_batch, gd_opt, args.task, wxh_updates)

if __name__=='__main__':

    main(sys.argv)

