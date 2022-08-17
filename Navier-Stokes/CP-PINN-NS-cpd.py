import sys
import tensorflow as tf
import numpy as np
import scipy.io
import time
import argparse
import time
import pickle

np.random.seed(1234)
tf.set_random_seed(1234)

PATH = './'
Net_layer = [3] + [20] * 8 + [2] 
test_Number = ''

parser = argparse.ArgumentParser()
parser.add_argument('--adam', default=100000, type=int)
parser.add_argument('--lbfgs', default=50000, type=int)
parser.add_argument('--case', 
                    default='NS_1bp_balanced', type=str, 
                    choices=("NS_1bp_balanced", "NS_1bp_imbalanced", "NS_2bp_balanced"))

parser.add_argument('--lr', default=0.001, type=int)
parser.add_argument('--obs', default=700, type=int)

args = parser.parse_args()
#%%
###############################################################################
###############################################################################
class CP_PINN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers, l_t, l_x):

        X = np.concatenate([x, y, t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        
        self.u = u
        self.v = v

        self.l_t = l_t
        self.l_x = l_x

        self.gamma = tf.Variable(tf.truncated_normal([self.l_t, 1], stddev=0.07,dtype=tf.float64), dtype=tf.float64)
        self.lambda_2 = tf.cumsum(self.gamma[::-1])[::-1]
       
        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.y_tf   = tf.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.t_tf   = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])
        self.v_tf   = tf.placeholder(tf.float64, shape=[None, self.v.shape[1]]) 

        self.weights, self.biases, self.a = self.initialize_NN(layers)

      
        self.u_NN_pred, self.v_NN_pred, self.p_NN_pred, self.f_u_pred, self.f_v_pred = self.net_f(self.x_tf, self.y_tf, self.t_tf)
      
        self.loss_f = (tf.reduce_sum(tf.square(self.u_tf - self.u_NN_pred)) + tf.reduce_sum(tf.square(self.v_tf - self.v_NN_pred)))
        self.loss_s = tf.reduce_sum(tf.square(self.f_u_pred)) + tf.reduce_sum(tf.square(self.f_v_pred))

        t = np.arange(1, l_t)
        u = tf.constant(((self.l_t/(t*(self.l_t - t)))**.5 * 50 + 50), shape=[self.l_t - 1, 1], dtype = tf.float64)
        self.loss_l1 = tf.reduce_sum(tf.multiply(u, tf.nn.relu(self.gamma[:-1]))) + tf.reduce_sum(tf.multiply(u, tf.nn.relu(-self.gamma[:-1])))

        self.loss  = self.loss_f + self.loss_s + self.loss_l1

                   
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
    
    def net_f(self, x, y, t):
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases,  self.a)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
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

        f_u = u_t + (u*u_x + v*u_y) + p_x - tf.reshape(tf.multiply(tf.reshape(lambda_2, [self.l_t]), tf.reshape((u_xx + u_yy), [self.l_x, self.l_t])), [self.l_t*self.l_x, 1])
        f_v = v_t + (u*v_x + v*v_y) + p_y - tf.reshape(tf.multiply(tf.reshape(lambda_2, [self.l_t]), tf.reshape((v_xx + v_yy), [self.l_x, self.l_t])), [self.l_t*self.l_x, 1])
        
        return u, v, p, f_u, f_v
    
    def lbfgs_callback(self, loss_value_s, loss_value_f, loss_valuel1, gamma):
        str_print = ''.join(['Lossp: %.3e, Lossb: %.3e, LossL1: %.3e, Position: %d'])
        print(str_print % (loss_value_s, loss_value_f, loss_valuel1, np.argmax(np.abs(gamma[:-1]))))
        
###############################################################################
    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.u_tf: self.u, self.v_tf: self.v}
        
        total_time_train = 0
        min_loss         = 1e16
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
                lambda_value = self.sess.run(self.lambda_2, tf_dict)
                total_records.append(np.array([it, loss_value, lambda_value]))
               
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Lossp: %.3e, Lossb: %.3e, LossL1: %.3e, Time: %.2f, TrTime: %.4f, Position: %d'])
                print(str_print % (it, loss_value_s, loss_value_f, loss_valuel1, elapsed, elapsed_time_train, np.argmax(np.abs(self.gamma[:-1]))))
                start_time = time.time()
                
        error_u = 1
        error_records = [loss_value, error_u]

        self.optimizer.minimize(self.sess, feed_dict = tf_dict, fetches = [self.loss_s, self.loss_f, self.loss_l1, self.gamma], 
                                loss_callback = self.lbfgs_callback)

        loss_value = self.sess.run(self.loss, tf_dict)
        loss_value_f= self.sess.run(self.loss_f, tf_dict)
        loss_value_s= self.sess.run(self.loss_s, tf_dict)
        lambda_value = self.sess.run(self.lambda_2)

        total_records.append(np.array([it + args.lbfgs, loss_value, lambda_value]))


        return lambda_value, error_records, total_records, total_time_train


if __name__ == "__main__": 
    
    ###########################################################################    
    ###########################################################################
    ##############No need to use XT_f_train, we can also use XT_u_train########

    

    if args.case == 'PINN_NS_2bp_balanced':
        with open('data_Re_2_100_2.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open('data_Re_2_50.pkl', 'rb') as f:
            data = pickle.load(f)
    sample_mesh = np.random.choice(data['X_star'].shape[0], args.obs)
    X_star = data['X_star'][sample_mesh, :]# N x 2
    U_star = data['U_star'][sample_mesh] # N x 2 x T
    P_star = data['p_star'][sample_mesh] # N x T
    t_star = data['t'] # T x 1

    if args.case == "PINN_NS_1bp_imbalanced":
        U_star = U_star[:, :, :135]
        P_star = P_star[:, :, :135]
        t_star = t_star[:135]

    l_x = U_star.shape[0]
    l_t = t_star.shape[0]

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

    model = CP_PINN(x, y, t, u, v, Net_layer, l_t, l_x)
    
    total_record_total=[]
#%%
    lambda_value, error_record, total_record, total_time_train\
    = model.train(args.adam)
    total_record_total.append([total_record])
#%%

    np.savetxt('lambda_value_' + args.case, lambda_value.flatten(), delimiter=',')
    
    with open(''.join([PATH,str(args.case), str(test_Number),'_record.mat']), 'wb') as f:
        scipy.io.savemat(f, {'x_test'      : X_star})
        scipy.io.savemat(f, {'u_test'      : U_star})
        scipy.io.savemat(f, {'total'       : total_record})
        scipy.io.savemat(f, {'total_time_train'   : total_time_train})
