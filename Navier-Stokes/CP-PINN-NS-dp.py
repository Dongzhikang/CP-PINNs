import tensorflow as tf
import numpy as np
import time
import pickle
import scipy.io
from scipy.signal import find_peaks
import argparse

PATH = './'

parser = argparse.ArgumentParser()
parser.add_argument('--adam', default=100000 + 1, type=int)
parser.add_argument('--lbfgs', default=50000, type=int)
parser.add_argument('--case', 
                    default='NS_1bp_balanced', type=str, 
                    choices=("NS_1bp_balanced", "NS_1bp_imbalanced", "NS_2bp_balanced"))

parser.add_argument('--lr', default=0.001, type=int)
parser.add_argument('--obs', default=700, type=int)

args = parser.parse_args()

class CP_PINN:
    # Initialize the class
    def __init__(self,  x, y, t, u, v, layers):
        
        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        
        self.u = u
        self.v = v
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # Initialize parameters
        self.lambda_value = tf.Variable(tf.truncated_normal([1,], stddev=.07), dtype=tf.float32)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred))

        self.it_lbfgs = args.adam
        self.total_records    = []

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]], name = 'w'+str(l))
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name='b'+str(l))
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size, name):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name = name)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        lambda_value = self.lambda_value
        psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]  
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + (u*u_x + v*u_y) + p_x - lambda_value*(u_xx + u_yy) 
        f_v = v_t + (u*v_x + v*v_y) + p_y - lambda_value*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v

    def callback(self, loss, lambda_value):
        self.it_lbfgs += 1
        if self.it_lbfgs % 100 == 0:
            str_print = ''.join(['Loss: %.3e, Lambda: %.4f'])
            print(str_print % (loss, lambda_value))
            self.total_records.append(np.array([self.it_lbfgs, loss, lambda_value]))


    def train(self, nIter):       
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': args.lbfgs,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps}) 
        optimizer_Adam = tf.train.AdamOptimizer()
        train_op_Adam = optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_time_train = 0
        start_time       = time.time()


        for it in range(nIter):
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}
                    
            start_time_train = time.time()
            self.sess.run(train_op_Adam, tf_dict)
            elapsed_time_train = time.time() - start_time_train
            total_time_train = total_time_train + elapsed_time_train

            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_value = self.sess.run(self.lambda_value)
                self.total_records.append(np.array([it, loss_value, lambda_value]))   
            
            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_value = self.sess.run(self.lambda_value)
                    
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, LossL1: %.3e, Time: %.2f, TrTime: %.4f, Lambda: %.4f'])
                print(str_print % (it, loss_value, elapsed, elapsed_time_train, lambda_value))
                start_time = time.time()

        optimizer.minimize(self.sess,
                        feed_dict = tf_dict,
                        fetches = [self.loss, self.lambda_value],
                        loss_callback = self.callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        
        lambda_value = self.sess.run(self.lambda_value)
        self.total_records.append(np.array([self.it_lbfgs, loss_value, lambda_value]))


        return lambda_value, self.total_records, total_time_train

    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star


layers = [3, 20,20,20,20,20,20,20,20, 2]

# Load Data

if args.case == 'NS_2bp_balanced':
    with open('data_Re_2_100_2.pkl', 'rb') as f:
        data = pickle.load(f)
    l_t = data['t'].shape[0]
elif args.case == 'NS_1bp_balanced':
    with open('data_Re_2_50.pkl', 'rb') as f:
        data = pickle.load(f)
    l_t = data['t'].shape[0]
else:
    with open('data_Re_2_50.pkl', 'rb') as f:
        data = pickle.load(f)
    l_t = 135

N_train = 5000

lambda_value = lambda_value = np.loadtxt(''.join(['lambda_', str(args.case)]))
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)

candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))

def dp(candidates):
        res = []
        for i in range(len(candidates) - 1):
            for j in range(i+1, len(candidates)):
                if [[candidates[i], candidates[j]]] not in res:
                    res += [[candidates[i], candidates[j]]]
        return res
    
intervals = dp(candidates)


for interval in intervals:
    print(interval)
    case = args.case + '_' + str(interval)

    U_star = data['U_star'][:, :, interval[0]:interval[1]] # N x 2 x T
    P_star = data['p_star'][:, interval[0]:interval[1]] # N x T
    t_star = data['t'][interval[0]:interval[1]] # T x 1
    X_star = data['X_star']# N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T

    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T

    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1

    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)

    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    p_train = p[idx,:]

    # Training
    model = CP_PINN(x_train, y_train, t_train, u_train, v_train, layers)
    lambda_value, total_record, total_time_train = model.train(args.adam)
    u_pred, v_pred, p_pred = model.predict(x_train, y_train, t_train)

    with open(''.join([PATH,str(case),'_record.mat']), 'wb') as f:
            scipy.io.savemat(f, {'total'       : total_record})
            scipy.io.savemat(f, {'total_time_train'   : total_time_train})
            scipy.io.savemat(f, {'x'   : x_train})
            scipy.io.savemat(f, {'y'   : y_train})
            scipy.io.savemat(f, {'t'   : t_train})
            scipy.io.savemat(f, {'u_train'   : u_train})
            scipy.io.savemat(f, {'v_train'   : v_train})
            scipy.io.savemat(f, {'p_train'   : p_train})
            scipy.io.savemat(f, {'u_pred'   : u_pred})
            scipy.io.savemat(f, {'v_pred'   : v_pred})
            scipy.io.savemat(f, {'p_pred'   : p_pred})
