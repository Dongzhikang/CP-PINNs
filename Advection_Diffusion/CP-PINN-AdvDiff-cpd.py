import sys
import tensorflow as tf
import numpy as np
import scipy.io
import time
import argparse

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################
PATH = './'
V = 1.0 
T = 1 

Net_layer = [2] + [5] * 3 + [1] 
test_Number = ''

parser = argparse.ArgumentParser()
parser.add_argument('--adam', default=100000, type=int)
parser.add_argument('--lbfgs', default=50000, type=int)
parser.add_argument('--case', 
                    default='ADE_1bp_balanced', type=str, 
                    choices=("ADE_1bp_balanced", "ADE_1bp_imbalanced", "ADE_2bp_balanced"))

parser.add_argument('--lr', default=0.001, type=int)

args = parser.parse_args()



###############################################################################
###############################################################################
class CP_PINN:
    # Initialize the class
    def __init__(self, XT_u_train, u_train, layers, lb, ub, l_t, l_x):
        self.lb = lb
        self.ub = ub
        self.l_t = l_t
        self.l_x = l_x
        self.x   = XT_u_train[:,0:1]
        self.t   = XT_u_train[:,1:2]
        self.u   = u_train

        self.diff_epsilon = tf.Variable(tf.truncated_normal([self.l_t, 1], dtype=tf.float64), dtype=tf.float64)
        self.epsilon = tf.cumsum(self.diff_epsilon[::-1])[::-1]      
       
        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf   = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]]) 

        self.weights, self.biases, self.a = self.initialize_NN(layers)

      
        self.u_NN_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)
   
        self.loss_f = tf.reduce_sum(tf.square(self.u_tf - self.u_NN_pred))
        self.loss_s = tf.reduce_sum(tf.square(self.f_pred))
        ##### U-shape curve penalty
        t = np.arange(1, l_t)
        u = tf.constant(((self.l_t/(t*(self.l_t - t)))**.5 * 40 + 50), shape=[self.l_t - 1, 1], dtype = tf.float64)
        self.loss_l1 = tf.reduce_sum(tf.multiply(u, tf.nn.relu(self.diff_epsilon[:-1]))) + tf.reduce_sum(tf.multiply(u, tf.nn.relu(-self.diff_epsilon[:-1])))
        self.loss  = self.loss_f +self.loss_s + self.loss_l1

                   
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': args.lbfgs,
                                                                                                      'maxfun': 50000,
                                                                                                      'maxcor': 50,
                                                                                                      'maxls': 50,
                                                                                                      'ftol': 1.0 * np.finfo(float).eps})


        self.LR = args.lr
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
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
    
    def lbfgs_callback(self, loss_value_s, loss_value_f, loss_valuel1, diff_epsilon):
        str_print = ''.join(['Loss_s: %.3e, Loss_f: %.3e, LossL1: %.3e, Position: %d'])
        print(str_print % (loss_value_s, loss_value_f, loss_valuel1, np.argmax(np.abs(diff_epsilon.flatten()[:-1]))))
        
###############################################################################
    def train(self, nIter):
        
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}
        
        total_time_train = 0
        start_time       = time.time()
        total_records    = []
        error_records    = []

        for it in range(nIter):
            
            start_time_train = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)
            elapsed_time_train = time.time() - start_time_train
            total_time_train = total_time_train + elapsed_time_train            
 
            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_value_f= self.sess.run(self.loss_f, tf_dict)
                loss_value_s= self.sess.run(self.loss_s, tf_dict)
                loss_valuel1 = self.sess.run(self.loss_l1, tf_dict)
                epsilon_value = self.sess.run(self.epsilon, tf_dict)
                diff_epsilon = self.sess.run(self.diff_epsilon, tf_dict)
                total_records.append(np.array([it, loss_value, np.mean(epsilon_value)]))
                
               
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Loss_s: %.3e, Loss_f: %.3e, LossL1: %.3e, Time: %.2f, TrTime: %.4f, Position: %d'])
                print(str_print % (it, loss_value_s, loss_value_f, loss_valuel1, elapsed, elapsed_time_train, np.argmax(np.abs(diff_epsilon[:-1]))))
                start_time = time.time()
                
        error_u = 1
        error_records = [loss_value, error_u]

        self.optimizer.minimize(self.sess, feed_dict = tf_dict, fetches = [self.loss_s, self.loss_f, self.loss_l1, self.diff_epsilon], 
                                loss_callback = self.lbfgs_callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        loss_value_f= self.sess.run(self.loss_f, tf_dict)
        loss_value_s= self.sess.run(self.loss_s, tf_dict)
        loss_valuel1 = self.sess.run(self.loss_l1, tf_dict)
        epsilon_value = self.sess.run(self.epsilon)
        total_records.append(np.array([it + args.lbfgs, loss_value, np.mean(epsilon_value)]))


        return epsilon_value, error_records, total_records, total_time_train



###############################################################################
############## For breakpoint problems, we can generate two parts of data and concatenate them #######################


if __name__ == "__main__": 
    
    
    ###########################################################################    

    def u_initial(x, t):
        utemp = - np.sin(np.pi*x)
        return utemp
  
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
        if args.case == 'ADE_1bp_balanced':
            if j < len(ttest) // 2: e = 1
            else: e = 0.05
        elif args.case == 'ADE_2bp_balanced':
            if j < len(ttest) // 3: e = 1
            elif j >= len(ttest) // 3  and j < 2*len(ttest) // 3: e = 0.05
            else: e = 1
        else:
            if j < len(ttest) // 5: e = 1
            else: e = 0.05
        data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
    data_temp = np.asarray(data_temp)

    ttest=ttest[:,None]
    xtest=xtest[:,None]
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    Exact = data_temp.flatten()[2::3]
    XT_test = np.hstack((Xtest[:,None],Ytest[:,None])) # dim: TN x 2
    u_test = Exact[:,None] # dim: TN x 1

    lb = XT_test.min(0)
    ub = XT_test.max(0)

    model = CP_PINN(XT_test, u_test, Net_layer, lb, ub, l_t, l_x)
    
    total_record_total=[]
#%%
    epsilon_value, error_record, total_record, total_time_train\
    = model.train(args.adam)
    total_record_total.append([total_record])
#%%

    np.savetxt('epsilon_value_' + args.case, epsilon_value.flatten(), delimiter=',')
    
    with open(''.join([PATH,str(args.case), str(test_Number),'_record.mat']), 'wb') as f:
        scipy.io.savemat(f, {'x_test'      : XT_test})
        scipy.io.savemat(f, {'u_test'      : u_test})
        scipy.io.savemat(f, {'total'       : total_record})
        scipy.io.savemat(f, {'total_time_train'   : total_time_train}) 

