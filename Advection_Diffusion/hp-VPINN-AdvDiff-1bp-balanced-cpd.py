
"""
@Title:
    hp-VPINNs: A General Framework For Solving PDEs
    Application to Advection Diffusion Eqn
@author: 
    Ehsan Kharazmi
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu
"""


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from pyDOE import lhs
import scipy.io
from scipy.special import legendre
from torch import lt
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################
PATH = './'
case = 'hpPINN_ADE_1bp_balanced'


LR = 0.001
Opt_Niter = 100000 + 1
epoch_lbfgs = 50000
Opt_tresh = 2e-11
var_form  = 1


#gamma = 2*np.pi
#epsilon = gamma/(np.pi)
V = 1.0 # constant coefficients?
T = 1 # total time length?

Net_layer = [2] + [5] * 3 + [1] 
N_el_x = 3 ## Number of domain decompositions x axis
N_el_t = 3 ## Number of domain decomposition t axis
N_test_x = N_el_x*[5] #Number of test functions
N_test_t = N_el_t*[5]
N_quad = 50 #10 Gauss quadreture
N_bound = 400 #80 # boundary points, useless

test_Number = ''
#%%
###############################################################################
###############################################################################
class VPINN:
    # Initialize the class
    def __init__(self, XT_u_train, u_train, XT_quad, W_quad,\
                 T_quad, WT_quad, grid_x, grid_t, N_testfcn, XT_test, u_test, layers, lb, ub, l_t, l_x):
        self.lb = lb
        self.ub = ub
        self.l_t = l_t
        self.l_x = l_x
        self.x    = XT_u_train[:,0:1]
        self.t    = XT_u_train[:,1:2]
        self.u    = u_train


        #self.diff_epsilon = tf.Variable(self.l_t*tf.ones([0.7], dtype=tf.float64), dtype=tf.float64)
        self.diff_epsilon = tf.Variable(tf.truncated_normal([self.l_t, 1], dtype=tf.float64), dtype=tf.float64)
        self.epsilon = tf.cumsum(self.diff_epsilon[::-1])[::-1]

        self.xquad  = XT_quad[:,0:1]
        self.tquad  = XT_quad[:,1:2]
        self.wquad  = W_quad

        self.tquad_1d = T_quad[:,None]
        self.xquad_1d = self.tquad_1d
        self.wquad_1d = WT_quad[:,None]
        
        self.xtest  = XT_test[:,0:1]
        self.ttest  = XT_test[:,1:2]
        self.utest  = u_test
        
        self.Nelementx = np.size(N_testfcn[0])
        self.Nelementt = np.size(N_testfcn[1])
        
       
        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf   = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]]) 


        self.x_test = tf.placeholder(tf.float64, shape=[None, self.xtest.shape[1]])
        self.t_test = tf.placeholder(tf.float64, shape=[None, self.ttest.shape[1]])
        self.x_quad = tf.placeholder(tf.float64, shape=[None, self.xquad.shape[1]])
        self.t_quad = tf.placeholder(tf.float64, shape=[None, self.tquad.shape[1]])

        self.weights, self.biases, self.a = self.initialize_NN(layers)

      
        self.u_NN_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)
        self.u_NN_test = self.net_u(self.x_test, self.t_test)

    
        self.varloss_total = 0

        for ex in range(self.Nelementx):
            for et in range(self.Nelementt):
                print('####################################################')
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementt = N_testfcn[1][et]

                jacobian       = (grid_t[et+1]-grid_t[et])/2*(grid_x[ex+1]-grid_x[ex])/2
                jacobian_x     = (grid_x[ex+1]-grid_x[ex])/2
                jacobian_t     = (grid_t[et+1]-grid_t[et])/2

                ##### 2D Integral evaluations  #####
                x_quad_element = tf.constant(grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(self.xquad+1), dtype=tf.float64)###?? why?
                t_quad_element = tf.constant(grid_t[et] + (grid_t[et+1]-grid_t[et])/2*(self.tquad+1), dtype=tf.float64)

                u_NN_quad_element = self.net_u(x_quad_element, t_quad_element)

                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, t_quad_element)
                d1tu_NN_quad_element = self.net_dtu(x_quad_element, t_quad_element)
                
                testx_quad_element = self.Test_fcn(Ntest_elementx, self.xquad)
                d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                testt_quad_element = self.Test_fcn(Ntest_elementt, self.tquad)
                d1testt_quad_element, d2testt_quad_element = self.dTest_fcn(Ntest_elementt, self.tquad)

                ##### 1D Integral evaluations for boundary terms in integration-by-parts #####
                t_quad_1d_element = tf.constant(grid_t[et] + (grid_t[et+1]-grid_t[et])/2*(self.tquad_1d+1), dtype=tf.float64)
                x_quad_1d_element = tf.constant(grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(self.xquad_1d+1), dtype=tf.float64)
#                x_b_element    = tf.constant(np.array([[grid_x[ex]], [grid_x[ex+1]]]))
                x_bl_element    = tf.constant(grid_x[ex], dtype=tf.float64, shape=t_quad_1d_element.shape) 
                x_br_element    = tf.constant(grid_x[ex+1], dtype=tf.float64, shape=t_quad_1d_element.shape) 
                t_tl_element    = tf.constant(grid_t[et], dtype=tf.float64, shape=x_quad_1d_element.shape) 
                t_tr_element    = tf.constant(grid_t[et+1], dtype=tf.float64, shape=x_quad_1d_element.shape) 

                u_NN_br_element = self.net_u(x_br_element, t_quad_1d_element)
                d1u_NN_br_element, d2u_NN_br_element = self.net_dxu(x_br_element, t_quad_1d_element)
                u_NN_bl_element = self.net_u(x_bl_element, t_quad_1d_element)
                d1u_NN_bl_element, d2u_NN_bl_element = self.net_dxu(x_bl_element, t_quad_1d_element)
                
                u_NN_tr_element = self.net_u(x_quad_1d_element, t_tr_element)
                u_NN_tl_element = self.net_u(x_quad_1d_element, t_tl_element)
                
                testx_bound_element = self.Test_fcn(Ntest_elementx, np.array([[-1],[1]]))
                d1testx_bound_element, d2testx_bounda_element = self.dTest_fcn(Ntest_elementx, np.array([[-1],[1]]))
                testt_quad_1d_element = self.Test_fcn(Ntest_elementt, self.tquad_1d)
                
                testt_bound_element = self.Test_fcn(Ntest_elementt, np.array([[-1],[1]]))
                testx_quad_1d_element = self.Test_fcn(Ntest_elementx, self.xquad_1d)
                

                integrand_part1 = d1tu_NN_quad_element 
                integrand_part1_VF22 = 0               

                
                if var_form == 0:
                    U_NN_element = tf.convert_to_tensor([[\
                                    jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]\
                                    *(d1tu_NN_quad_element + V*d1xu_NN_quad_element - self.epsilon*d2xu_NN_quad_element))\
#                                    *(d1tu_NN_quad_element - self.epsilon*d2xu_NN_quad_element))\
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype= tf.float64)
                        
                if var_form == 1:

                    U_NN_element = tf.convert_to_tensor([[\
                         jacobian*tf.reduce_sum(self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]*(d1tu_NN_quad_element + V*d1xu_NN_quad_element))\
                       + self.epsilon*jacobian/jacobian_x*tf.reduce_sum(self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]*d1xu_NN_quad_element)\
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype= tf.float64)
                    



   
                Res_NN_element = tf.reshape(U_NN_element , [1,-1])
                loss_element = tf.reduce_sum(tf.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element
      
        #self.lossb = 10*tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossb = tf.reduce_sum(tf.square(self.u_tf - self.u_NN_pred))
        self.lossv = self.varloss_total
        self.lossp = tf.reduce_sum(tf.square(self.f_pred))
        ##### U-shape curve penalty?
        t = np.arange(1, l_t)
        u = tf.constant(((self.l_t/(t*(self.l_t - t)))**.5 * 40 + 50), shape=[self.l_t - 1, 1], dtype = tf.float64)
        self.loss_l1 = tf.reduce_sum(tf.multiply(u, tf.nn.relu(self.diff_epsilon[:-1]))) + tf.reduce_sum(tf.multiply(u, tf.nn.relu(-self.diff_epsilon[:-1])))
        self.loss  = self.lossb +self.lossp + self.loss_l1 # + self.varloss_b_total#+ 0.01*loss_l2

                   
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': epoch_lbfgs,
                                                                                                      'maxfun': 50000,
                                                                                                      'maxcor': 50,
                                                                                                      'maxls': 50,
                                                                                                      'ftol': 1.0 * np.finfo(float).eps})


        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
#        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


###############################################################################
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.01, dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases, a
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim), dtype=np.float64)
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)
 
    
    def neural_net(self, X, weights, biases, a):
        num_layers = len(weights) + 1
        H = X 
        # H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases,  self.a)
        return u

    def net_dxu(self, x, t):
        u   = self.net_u(x, t)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu, d2xu
    
    def net_dtu(self, x, t):
        u   = self.net_u(x, t)
        d1tu = tf.gradients(u, t)[0]
        return d1tu

    def net_f(self, x, t):
        u = self.net_u(x,t)
        d1tu = tf.gradients(u, t)[0]
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        f = d1tu + V*d1xu - tf.reshape(self.diff_epsilon*tf.reshape(d2xu, [self.l_t, self.l_x]), [self.l_t*self.l_x, 1])
        return f



    def Test_fcn(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            test_total.append(test)
        return np.asarray(test_total)

## Legendre Test
    def dTest_fcn(self, N_test,x):
        d1test_total = []
        d2test_total = []
        for n in range(1,N_test+1):
            if n==1:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n==2:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
            else:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
        return np.asarray(d1test_total), np.asarray(d2test_total)


    def callback(self, lossv, lossb):
        print('Lossv: %e, Lossb: %e' % (lossv, lossb))
    
    def lbfgs_callback(self, loss_valuev, loss_valuep, loss_valueb, loss_valuel1, diff_epsilon):
        str_print = ''.join(['Lossv: %.3e, Lossp: %.3e, Lossb: %.3e, LossL1: %.3e, Position: %d'])
        print(str_print % (loss_valuev, loss_valuep, loss_valueb, loss_valuel1, np.argmax(np.abs(diff_epsilon.flatten()[:-1]))))
        #print(str_print % (loss_valuev, loss_valuep, loss_valueb, loss_valuel1, np.mean(diff_epsilon.flatten())))
        
###############################################################################
    def train(self, nIter, tresh):
        
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,\
                   self.x_quad: self.xquad, self.t_quad: self.tquad,\
                   self.x_test: self.xtest, self.t_test: self.ttest}
        
        total_time_train = 0
        min_loss         = 1e16
        start_time       = time.time()
        total_records    = []
        error_records    = []
        #u_records_iterhis   = []

        for it in range(nIter):
            
            start_time_train = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)
            elapsed_time_train = time.time() - start_time_train
            total_time_train = total_time_train + elapsed_time_train            
            
 
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueb= self.sess.run(self.lossb, tf_dict)
                loss_valuev= self.sess.run(self.lossv, tf_dict)
                #loss_valuep= 1 
                loss_valuep= self.sess.run(self.lossp, tf_dict)
                loss_valuel1 = self.sess.run(self.loss_l1, tf_dict)
                epsilon_value = self.sess.run(self.epsilon, tf_dict)
                diff_epsilon = self.sess.run(self.diff_epsilon, tf_dict)
                a_value = 1 
                total_records.append(np.array([it, loss_value, np.mean(epsilon_value), a_value]))
                
                if loss_value < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    break

                if it > 0.9*nIter and loss_value < min_loss:
                    min_loss      = loss_value
                    u_pred     = self.sess.run(self.u_NN_test, tf_dict)
                    #u_records     = u_pred
               
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Lossv: %.3e, Lossp: %.3e, Lossb: %.3e, LossL1: %.3e, Time: %.2f, TrTime: %.4f, Position: %d'])
                print(str_print % (it, loss_valuev, loss_valuep, loss_valueb, loss_valuel1, elapsed, elapsed_time_train, np.argmax(np.abs(diff_epsilon[:-1]))))
                start_time = time.time()
                
        error_u = 1
        error_records = [loss_value, error_u]

        self.optimizer.minimize(self.sess, feed_dict = tf_dict, fetches = [self.lossv, self.lossp, self.lossb, self.loss_l1, self.diff_epsilon], 
                                loss_callback = self.lbfgs_callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        loss_valueb= self.sess.run(self.lossb, tf_dict)
        loss_valuev= self.sess.run(self.lossv, tf_dict)
        #loss_valuep= 1 
        loss_valuep= self.sess.run(self.lossp, tf_dict)
        loss_valuel1 = self.sess.run(self.loss_l1, tf_dict)
        epsilon_value = self.sess.run(self.epsilon)
        a_value = 1 
        total_records.append(np.array([it + epoch_lbfgs, loss_value, np.mean(epsilon_value), a_value]))


        return epsilon_value, error_records, total_records, total_time_train



###############################################################################
############## For breakpoint problems, we can generate two parts of data and concatenate them #######################


if __name__ == "__main__": 
    
    ###########################################################################    

    def u_initial(x, t):
        utemp = - np.sin(np.pi*x)
        return utemp

    ###########################################################################
    # Quadrature points
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0) # Quadrature points on x
    T_quad, WT_quad   = (X_quad, WX_quad) # Quadrature points on t, why they are the same?
    xx, tt            = np.meshgrid(X_quad,  T_quad)
    ## What is wxx and wtt??? weight?
    wxx, wtt          = np.meshgrid(WX_quad, WT_quad)
    XT_quad_train     = np.hstack((xx.flatten()[:,None],  tt.flatten()[:,None]))
    WXT_quad_train    = np.hstack((wxx.flatten()[:,None], wtt.flatten()[:,None]))

    ###########################################################################
    NE_x, NE_t = N_el_x, N_el_t
    [x_l, x_r] = [-1, 1]
    [t_i, t_f] = [0, T]
    delta_x    = (x_r - x_l)/NE_x
    delta_t    = (t_f - t_i)/NE_t
    grid_x     = np.asarray([ x_l + i*delta_x for i in range(NE_x+1)])
    grid_t     = np.asarray([ t_i + i*delta_t for i in range(NE_t+1)])

    N_testfcn_total = [N_test_x, N_test_t]

 
#%%    
    ###########################################################################
    def u_ext(x, t, epsilon, trunc = 800):
        """
        Function to compute the analytical solution as a Fourier series expansion.
        Inputs:
            x: column vector of locations
            t: column vector of times
            trunc: truncation number of Fourier bases
        """

        # Series index:
        p = np.arange(0, trunc+1.0)
        p = np.reshape(p, [1, trunc+1])
        
        D = epsilon
        c0 = 16*np.pi**2*D**3*V*np.exp(V/D/2*(x-V*t/2))                           # constant
        
        c1_n = (-1)**p*2*p*np.sin(p*np.pi*x)*np.exp(-D*p**2*np.pi**2*t)           # numerator of first component
        c1_d = V**4 + 8*(V*np.pi*D)**2*(p**2+1) + 16*(np.pi*D)**4*(p**2-1)**2     # denominator of first component
        c1 = np.sinh(V/D/2)*np.sum(c1_n/c1_d, axis=-1, keepdims=True)             # first component of the solution
        
        c2_n = (-1)**p*(2*p+1)*np.cos((p+0.5)*np.pi*x)*np.exp(-D*(2*p+1)**2*np.pi**2*t/4)
        c2_d = V**4 + (V*np.pi*D)**2*(8*p**2+8*p+10) + (np.pi*D)**4*(4*p**2+4*p-3)**2
        c2 = np.cosh(V/D/2)*np.sum(c2_n/c2_d, axis=-1, keepdims=True)       # second component of the solution
        
        c = c0*(c1+c2)

        
        
        if t==0:
            c = u_initial(x, t)
        
        return c


    delta_test = 0.01 # time step
    xtest = np.linspace(-1,1,500) 
    ttest = np.arange(0, T+delta_test, delta_test)

    l_t = len(ttest)
    l_x = len(xtest)

    #data_temp = np.asarray([[ [xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 1.5)] for i in range(len(xtest))] for j in range(len(ttest))])
    ###############One change point#####################
    data_temp = []
    for j in range(len(ttest)):
        if j < len(ttest) // 2: e = 1
        else: e = 0.05
        data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
    data_temp = np.asarray(data_temp)

    #data_temp = data_temp[l_t//2+1:, :, :]
    #l_t = l_t // 2

    ttest=ttest[:,None]
    xtest=xtest[:,None]
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    Exact = data_temp.flatten()[2::3]
    XT_test = np.hstack((Xtest[:,None],Ytest[:,None])) # dim: TN x 2
    u_test = Exact[:,None] # dim: TN x 1

    lb = XT_test.min(0)
    ub = XT_test.max(0)

    #### Interior training points for inverse problem
    '''NPu_inter = 5
    x_inter_1 = np.empty(NPu_inter)[:,None]
    x_inter_1.fill(-0.5)
    t_inter_1 = T*lhs(1,NPu_inter)

    x_inter_2 = np.empty(NPu_inter)[:,None]
    x_inter_2.fill(0.0)
    t_inter_2 = T*lhs(1,NPu_inter)
    
    x_inter_3 = np.empty(NPu_inter)[:,None]
    x_inter_3.fill(0.5)
    t_inter_3 = T*lhs(1,NPu_inter)

    xu_inter = np.concatenate([x_inter_1, x_inter_2, x_inter_3]) ## interior x points 5*3 = 15 interior points
    timeu_inter = np.concatenate([t_inter_1, t_inter_2, t_inter_3]) ## interior t points
    XT_u_inter_train = np.hstack((xu_inter,timeu_inter))
    ## Find interior solution u
    u_inter_train = np.asarray([ u_ext(XT_u_inter_train[i,0],XT_u_inter_train[i,1]) for i in range(XT_u_inter_train.shape[0])]).flatten()[:,None]

    XT_u_train = np.concatenate((x_up_train, x_lo_train, x_in_train, XT_u_inter_train)) ## (80+80+80+15) x 2
    u_train = np.concatenate((u_up_train, u_lo_train, u_in_train, u_inter_train))''' ## (80+80+80+15) x 1


#%%
    ###########################################################################
    ##############No need to use XT_f_train, we can also use XT_u_train########

    model = VPINN(XT_test, u_test, XT_quad_train, WXT_quad_train,\
                  T_quad, WT_quad, grid_x, grid_t, N_testfcn_total, XT_test, u_test, Net_layer, lb, ub, l_t, l_x)
    
    total_record_total=[]
#%%
    epsilon_value, error_record, total_record, total_time_train\
    = model.train(Opt_Niter, Opt_tresh)
    total_record_total.append([total_record])
#%%

    '''np.savetxt('epsilon_value_1bp_balanced', epsilon_value.flatten(), delimiter=',')
    
    with open(''.join([PATH,str(case), str(test_Number),'_record.mat']), 'wb') as f:
        scipy.io.savemat(f, {'x_test'      : XT_test})
        scipy.io.savemat(f, {'u_test'      : u_test})
        scipy.io.savemat(f, {'grid_x'      : grid_x})
        scipy.io.savemat(f, {'grid_t'      : grid_t})
        #scipy.io.savemat(f, {'u_pred'      : u_record})
        #scipy.io.savemat(f, {'u_pred_his'  : u_records_iterhis})
        scipy.io.savemat(f, {'total'       : total_record})
        scipy.io.savemat(f, {'total_time_train'   : total_time_train})'''
    

    
    
    

