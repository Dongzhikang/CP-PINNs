import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker, cm
import scipy.io
from scipy.signal import find_peaks
import pickle
from scipy import stats
from scipy.stats import mstats
import pandas as pd
import argparse


np.random.seed(1234)
###############################################################################
###############################################################################
PATH = './'

test_Number = ''
fontsize = 24

parser = argparse.ArgumentParser()
parser.add_argument('--adam', default=100000, type=int)
parser.add_argument('--lbfgs', default=50000, type=int)
parser.add_argument('--case', 
                    default='NS_1bp_balanced', type=str, 
                    choices=("NS_1bp_balanced", "NS_1bp_imbalanced", "NS_2bp_balanced"))

parser.add_argument('--type', 
                    default='cpd', type=str, 
                    choices=("cpd", "his_coeff", "his_loss", "error"))

args = parser.parse_args()
print(args.case)
lambda_value = np.loadtxt(''.join(['lambda_', str(args.case)]))
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
l_t = len(lambda_value)
candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))

def dp(candidates):
        res = []
        for i in range(len(candidates) - 1):
            for j in range(i+1, len(candidates)):
                if [[candidates[i], candidates[j]]] not in res:
                    res += [[candidates[i], candidates[j]]]
        return res
    
intervals = dp(candidates)

if args.type == 'cpd':
    fig = plt.figure(1)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$Time$', fontsize = fontsize)
    plt.ylabel('$ log((|\Delta\lambda(t)|)^{-1})$', fontsize = fontsize)
    plt.grid(True)
    plt.plot(np.log(1/np.abs(np.diff(lambda_value.flatten()))))
    plt.plot(peaks, np.log(1/np.abs(np.diff(lambda_value.flatten())))[peaks], "x", color = 'red', label = 'Detection')
    plt.axvline(x=100, color='b', linestyle='--', label = ' Exact breakpoint')
    if args.case == 'NS_2bp_balanced':
        plt.axvline(x=200, color='b', linestyle='--', label = ' Exact breakpoint')
    
    legend = plt.legend(shadow=True, loc='best', fontsize=18)
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=6)    

    plt.savefig(''.join([PATH,str(args.case),'_cpd','.pdf'])) 




if args.type == 'his_coeff':

    for interval in intervals:
        case = args.case + '_' + str(interval) 
        records = scipy.io.loadmat(''.join([str(case), '_record.mat']))
        iteration = np.asarray(records['total'])[:,0]
        diff_his  = np.asarray(records['total'])[:,2]

        fig = plt.figure(2)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize = fontsize)
        plt.ylabel('$diffusion \,\, coeff. \,\, (\lambda)$', fontsize = fontsize)
        plt.grid(True)
        plt.plot(iteration, diff_his, linewidth=2, label = str(interval))

    plt.axhline(0.5, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_1$,$\lambda_3$')
    plt.axhline(0.01, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_2$')
    if args.case == 'NS_2bp_balanced':
        legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 1, bbox_to_anchor=(1.08, 1))
    else:
        legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 2)
    plt.xticks(np.array([0, 50000, 100000, 150000]))
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=6)
    plt.savefig(''.join([args.case,'_diffcoeff','.pdf']), bbox_inches='tight')     

if args.type == 'his_loss':
    for interval in intervals:
        case = args.case + '_' + str(interval)
        records = scipy.io.loadmat(''.join([PATH,str(case), '_record.mat']))

        iteration = np.asarray(records['total'])[:,0]
        loss_his  = np.asarray(records['total'])[:,1]

        fig = plt.figure(3)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize = fontsize)
        plt.ylabel('$loss \,\, values$', fontsize = fontsize)
        plt.yscale('log')
        plt.grid(True)
        plt.plot(iteration,loss_his, label = str(interval))
    if args.case == 'NS_2bp_balanced':
        legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 1, bbox_to_anchor=(1.08, 1))
    else:
        legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 3)
    plt.xticks(np.array([0, 50000, 100000, 150000]))
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=6) 
    plt.savefig(''.join([args.case,'_loss','.pdf']), bbox_inches='tight')   

if args.type == 'error':
    mse_u = []
    mse_v = []
    mpe_lambda = []
    d = {}
    if args.case == 'NS_2bp_balanced':
        real_lambda = np.asarray([.5]*100 + [0.01]*100 + [0.5]*100)
    elif args.case == 'NS_1bp_balanced':
        real_lambda = np.asarray([.5]*100 + [0.02]*100)
    else:
        real_lambda = np.asarray([.5]*100 + [0.02]*35)
    l_t = len(real_lambda)

    intervals = dp(candidates)
    
    d['Interval'] = [tuple(i) for i in intervals]
    for interval in intervals:
        case = args.case + '_' + str(interval)
        records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

        diff_his  = np.asarray(records['total'])[:,2]
        loss_his  = np.asarray(records['total'])[:,1]

        records = scipy.io.loadmat(''.join([PATH,case, '_record.mat']))
        mse_u.append(np.mean((records['u_train'] - records['u_pred'])**2))
        mse_v.append(np.mean((records['v_train'] - records['v_pred'])**2))
        mpe_lambda.append(np.mean(np.abs(diff_his[-1] - real_lambda[interval[0]:interval[1]]) / real_lambda[interval[0]:interval[1]])*100)

    d['MSE of u'] = mse_u
    d['MSE of v'] = mse_v
    d['MPE of $\lambda$'] = mpe_lambda
    d = pd.DataFrame(d)
    print(d)


'''with open('data_Re_2_50.pkl', 'rb') as f:
    data = pickle.load(f)

lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_balanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)

l_t = len(lambda_value)
case = 'PINN_NS_1bp_balanced'


records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

iteration = np.asarray(records['total'])[:,0]
diff_his  = np.asarray(records['total'])[:,2]
lambda_value = diff_his[-1]

fig = plt.figure(1)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
plt.xlabel('$Time$', fontsize = fontsize)
plt.ylabel('$ log((|\Delta\lambda(t)|)^{-1})$', fontsize = fontsize)
plt.grid(True)
#plt.plot(epsilon_value, 'p')
#plt.plot(np.log(1/np.abs(np.diff(epsilon_value.flatten()))))
plt.plot(np.log(1/np.abs(np.diff(lambda_value.flatten()))))
#plt.axvline(x=np.argmax(np.abs(np.diff(epsilon_value.flatten()))), color='y', linestyle='--', label = 'Detection')
print(peaks)
plt.plot(peaks, np.log(1/np.abs(np.diff(lambda_value.flatten())))[peaks], "x", color = 'red', label = 'Detection')
plt.axvline(x=100, color='b', linestyle='--', label = ' Exact breakpoint')
#plt.axvline(x=200, color='b', linestyle='--', label = ' Exact breakpoint')
legend = plt.legend(shadow=True, loc='best', fontsize=18)
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6)
plt.savefig(''.join([PATH,str(case),str(test_Number),'_cpd','.pdf'])) 
lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_balanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
l_t = data['t'].shape[0]
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
    
    case = 'PINN_NS_1bp_balanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    diff_his  = np.asarray(records['total'])[:,2]


    fig = plt.figure(2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$diffusion \,\, coeff. \,\, (\lambda)$', fontsize = fontsize)
    plt.grid(True)
    plt.plot(iteration, diff_his, linewidth=2, label = str(interval))

plt.axhline(0.5, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_1$')
plt.axhline(0.02, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_2$')

legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 2)
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6)
#plt.show()
plt.savefig(''.join(['PINN_NS_1bp_balanced','_diffcoeff','.pdf']))     

for interval in intervals:
    case = 'PINN_NS_1bp_balanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    loss_his  = np.asarray(records['total'])[:,1]


    fig = plt.figure(3)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(iteration,loss_his, label = str(interval))
legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 3)
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6) 
plt.savefig(''.join(['PINN_NS_1bp_balanced','_loss','.pdf']))
'''



############################################################################3
################################################################################
############################### ns 1bp balanced 3D contour error - U###################################




'''from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, cm

def ns_1_bp_balanced_error_scatter(interval, axis):
    case = 'PINN_NS_1bp_balanced_' + str(interval)
    records = scipy.io.loadmat(''.join([PATH,case, '_record.mat']))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('$x$', fontsize = fontsize)
    ax.set_ylabel('$y$', fontsize = fontsize)
    ax.set_zlabel('$t$', fontsize = fontsize)

    x = records['x']
    y = records['y']
    z = records['t']
    if axis == 'u':
        c = records['u_train'] - records['u_pred']
    if axis == 'v':
        c = records['v_train'] - records['v_pred']
    if axis == 'p':
        c = records['p_train'] - records['p_pred']
    fig.set_size_inches(w=11,h=11) 
    #plt.tick_params( labelsize = 20)
    plt.xticks(np.array([-5, -2, 1, 4, 7, 10]))
    plt.yticks(np.array([-5, -3, -1, 1, 3, 5]))
    img = ax.scatter(x, y, z, c=c, cmap=cm.jet)
    cbar = fig.colorbar(img, shrink=1)
    cbar.ax.tick_params(labelsize = 26)
    plt.savefig(''.join([PATH,str(case),'_',axis ,'_error','.png']), dpi = 400)

lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_balanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
l_t = data['t'].shape[0]
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
    ns_1_bp_balanced_error_scatter(interval, 'u')
    ns_1_bp_balanced_error_scatter(interval, 'v')
    ns_1_bp_balanced_error_scatter(interval, 'p')'''


##########################################################3
############################################################
############ NS 2 bp diff_coeff and loss ##################
##########################################################
#############################################################


'''with open('data_Re_2_100_2.pkl', 'rb') as f:
    data = pickle.load(f)

lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_2bp_balanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
l_t = data['t'].shape[0]

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
    
    case = 'PINN_NS_2bp_balanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    diff_his  = np.asarray(records['total'])[:,2]


    fig = plt.figure(2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$diffusion \,\, coeff. \,\, (\lambda)$', fontsize = fontsize)
    plt.grid(True)
    plt.plot(iteration, diff_his, linewidth=2, label = str(interval))

plt.axhline(0.5, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_1$,$\lambda_3$')
plt.axhline(0.01, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_2$')

legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 1, bbox_to_anchor=(1.08, 1))
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6)
#plt.show()
plt.savefig(''.join(['PINN_NS_2bp_balanced','_diffcoeff','.pdf']),  bbox_inches='tight')    '''

'''
for interval in intervals:
    case = 'PINN_NS_2bp_balanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    loss_his  = np.asarray(records['total'])[:,1]


    fig = plt.figure(3)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(iteration,loss_his, label = str(interval))
legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 1, bbox_to_anchor=(1.08, 1))
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6) 
plt.savefig(''.join(['PINN_NS_2bp_balanced','_loss','.pdf']),  bbox_inches='tight')'''


#############################################################################33
##############################################################################33
##############################################################################
################# ns 2bp balanced 3D error scatter ##########################
###########################################################################
########################################################################

'''import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, cm
with open('data_Re_2_100_2.pkl', 'rb') as f:
    data = pickle.load(f)
def ns_2_bp_balanced_error_scatter(interval, axis):
    print(interval)
    case = 'PINN_NS_2bp_balanced_' + str(interval)
    records = scipy.io.loadmat(''.join([PATH,case, '_record.mat']))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('$x$', fontsize = fontsize)
    ax.set_ylabel('$y$', fontsize = fontsize)
    ax.set_zlabel('$t$', fontsize = fontsize)

    x = records['x']
    y = records['y']
    z = records['t']
    if axis == 'u':
        c = records['u_train'] - records['u_pred']
    if axis == 'v':
        c = records['v_train'] - records['v_pred']
    if axis == 'p':
        c = records['p_train'] - records['p_pred']
    fig.set_size_inches(w=11,h=11) 
    #plt.tick_params( labelsize = 20)
    plt.xticks(np.array([-5, -2, 1, 4, 7, 10]))
    plt.yticks(np.array([-5, -3, -1, 1, 3, 5]))
    img = ax.scatter(x, y, z, c=c, cmap=cm.jet)
    cbar = fig.colorbar(img, shrink=1)
    cbar.ax.tick_params(labelsize = 26)
    plt.savefig(''.join([PATH,str(case),'_',axis ,'_error','.png']), dpi = 400)

lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_2bp_balanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
l_t = data['t'].shape[0]
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
    ns_2_bp_balanced_error_scatter(interval, 'u')
    ns_2_bp_balanced_error_scatter(interval, 'v')
    ns_2_bp_balanced_error_scatter(interval, 'p')'''




###########################################################################
################### 1bp imbalanced ####################################33
#########################################################################

'''with open('data_Re_2_50.pkl', 'rb') as f:
    data = pickle.load(f)

lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_imbalanced')
peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)

l_t = len(lambda_value)
case = 'PINN_NS_1bp_imbalanced'

fig = plt.figure(1)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
plt.xlabel('$Time$', fontsize = fontsize)
plt.ylabel('$ log((|\Delta\lambda(t)|)^{-1})$', fontsize = fontsize)
plt.grid(True)
plt.plot(np.log(1/np.abs(np.diff(lambda_value.flatten()))))
print(peaks)
plt.plot(peaks, np.log(1/np.abs(np.diff(lambda_value.flatten())))[peaks], "x", color = 'red', label = 'Detection')
plt.axvline(x=100, color='b', linestyle='--', label = ' Exact breakpoint')
legend = plt.legend(shadow=True, loc='best', fontsize=18)
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6)
#plt.show()
plt.savefig(''.join([PATH,str(case),str(test_Number),'_cpd','.pdf'])) 


l_t = 135
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
    
    case = 'PINN_NS_1bp_imbalanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    diff_his  = np.asarray(records['total'])[:,2]


    fig = plt.figure(2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$diffusion \,\, coeff. \,\, (\lambda)$', fontsize = fontsize)
    plt.grid(True)
    plt.plot(iteration, diff_his, linewidth=2, label = str(interval))

plt.axhline(0.5, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_1$')
plt.axhline(0.02, linewidth=2, linestyle='--', color='black', label = 'Exact $\lambda_2$')

legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 2)
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6)
#plt.show()
plt.savefig(''.join(['PINN_NS_1bp_imbalanced','_diffcoeff','.pdf']))     

for interval in intervals:
    case = 'PINN_NS_1bp_imbalanced_'  + str(interval) 
    records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

    iteration = np.asarray(records['total'])[:,0]
    loss_his  = np.asarray(records['total'])[:,1]


    fig = plt.figure(3)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(iteration,loss_his, label = str(interval))
legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 3)
plt.xticks(np.array([0, 50000, 100000, 150000]))
plt.tick_params( labelsize = 20)
#fig.tight_layout()
fig.set_size_inches(w=11,h=6) 
plt.savefig(''.join(['PINN_NS_1bp_imbalanced','_loss','.pdf']))'''

'''import pandas as pd

def dp(candidates):
        res = []
        for i in range(len(candidates) - 1):
            for j in range(i+1, len(candidates)):
                if [[candidates[i], candidates[j]]] not in res:
                    res += [[candidates[i], candidates[j]]]
        return res

def print_mse(situation):
    mse_u = []
    mse_v = []
    mpe_lambda = []
    d = {}
    if situation == '2bp_balanced':
        with open('data_Re_2_100_2.pkl', 'rb') as f:
            data = pickle.load(f)

        lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_2bp_balanced')
        peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
        real_lambda = np.asarray([.5]*100 + [0.01]*100 + [0.5]*100)
        l_t = data['t'].shape[0]
        candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))

    elif situation == '1bp_balanced':
        with open('data_Re_2_50.pkl', 'rb') as f:
            data = pickle.load(f)
        lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_balanced')
        peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
        real_lambda = np.asarray([.5]*100 + [0.02]*100)
        l_t = data['t'].shape[0]
        candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))
    else:
        with open('data_Re_2_50.pkl', 'rb') as f:
            data = pickle.load(f)
        lambda_value = np.loadtxt('/home/darren/Documents/hp-VPINNs/lambda_ns_1bp_imbalanced')
        peaks, val = find_peaks(np.abs(np.diff(lambda_value.flatten())), height = 0.0055)
        real_lambda = np.asarray([.5]*100 + [0.02]*35)
        l_t = 135
        candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))

    intervals = dp(candidates)
    
    d['Interval'] = [tuple(i) for i in intervals]
    for interval in intervals:
        case = 'PINN_NS_' + situation + '_' + str(interval)
        records = scipy.io.loadmat(''.join([PATH,str(case), str(test_Number),'_record.mat']))

        diff_his  = np.asarray(records['total'])[:,2]
        loss_his  = np.asarray(records['total'])[:,1]
        true_lambda = mstats.mode(real_lambda[interval[0]:interval[1]])[0][0]
        if situation == '2bp_balanced':
            true_lambda = mstats.mode(real_lambda[interval[0]:interval[1]])[0][0]
        else:
            print(situation)
            true_lambda = real_lambda[interval[0]]
        print(interval, diff_his[-1], loss_his[-1])
        records = scipy.io.loadmat(''.join([PATH,case, '_record.mat']))
        mse_u.append(np.mean((records['u_train'] - records['u_pred'])**2))
        mse_v.append(np.mean((records['v_train'] - records['v_pred'])**2))
        #true_lambda  =np.median(real_lambda[interval[0]:interval[1]])
        #mpe_lambda.append((np.abs((true_lambda-diff_his[-1]))/true_lambda*100).flatten()[0])
        mpe_lambda.append(np.mean(np.abs(diff_his[-1] - real_lambda[interval[0]:interval[1]]) / real_lambda[interval[0]:interval[1]])*100)

    d['MSE of u'] = mse_u
    d['MSE of v'] = mse_v
    d['MPE of $\lambda$'] = mpe_lambda
    d = pd.DataFrame(d)
    #print(d.to_latex(index = False))

print_mse('2bp_balanced')
print_mse('1bp_balanced')
print_mse('1bp_imbalanced')'''