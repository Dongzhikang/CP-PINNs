import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker, cm
import scipy.io
from scipy.signal import find_peaks
import time
import argparse


np.random.seed(1234)
###############################################################################
###############################################################################
PATH = './'

test_Number = ''
fontsize = 24
V = 1.0



parser = argparse.ArgumentParser()
parser.add_argument('--adam', default=100000, type=int)
parser.add_argument('--lbfgs', default=50000, type=int)
parser.add_argument('--case', 
                    default='ADE_1bp_balanced', type=str, 
                    choices=("ADE_1bp_balanced", "ADE_1bp_imbalanced", "ADE_2bp_balanced"))

parser.add_argument('--type', 
                    default='cpd', type=str, 
                    choices=("cpd", "his_coeff", "his_loss", "exact", "estimated", "error"))

args = parser.parse_args()
print(args.case)


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
        c = - np.sin(np.pi*x)
    
    return c

def dp(candidates):
    res = []
    for i in range(len(candidates) - 1):
        for j in range(i+1, len(candidates)):
            if [[candidates[i], candidates[j]]] not in res:
                res += [[candidates[i], candidates[j]]]
    return res

epsilon_value = np.loadtxt(''.join(['epsilon_value_', str(args.case)]))
l_t = len(epsilon_value)

height = {'ADE_1bp_balanced': 0.002,
              'ADE_1bp_imbalanced': 1,
              'ADE_2bp_balanced': 0.00005}
peaks, val = find_peaks(np.abs(np.diff(epsilon_value.flatten())), height = height[args.case])
candidates = np.concatenate((np.array([0]), peaks, np.array([l_t])))
intervals = dp(candidates)


if args.type == 'cpd':
    fig = plt.figure(1)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$Time$', fontsize = fontsize)
    plt.ylabel('$ log((|\Delta\kappa(t)|)^{-1})$', fontsize = fontsize)
    plt.grid(True)
    plt.plot(np.log(1/np.abs(np.diff(epsilon_value.flatten()))))
    
    plt.plot(peaks, np.log(1/val['peak_heights']), "x", color = 'red', label = 'Detection')

    if args.case == 'ADE_1bp_balanced':
        plt.axvline(x=l_t // 2, color='b', linestyle='--', label = ' Exact breakpoint')
    elif args.case == 'ADE_2bp_balanced':
        plt.axvline(x=l_t // 3, color='b', linestyle='--', label = ' Exact breakpoint')
        plt.axvline(x=2*l_t // 3, color='b', linestyle='--')
    else:
        plt.axvline(x=l_t // 5, color='b', linestyle='--', label = ' Exact breakpoint')

    legend = plt.legend(shadow=True, loc='best', fontsize=18)
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=6)
    plt.savefig(''.join([PATH,str(args.case),'_cpd','.pdf'])) 


if args.type == 'his_coeff':
    for interval in intervals:
        case = args.case + '_'  + str(interval) 
        records = scipy.io.loadmat(''.join([str(case), '_record.mat']))
        iteration = np.asarray(records['total'])[:,0]
        diff_his  = np.asarray(records['total'])[:,2]

        fig = plt.figure(2)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize = fontsize)
        plt.ylabel('$diffusion \,\, coeff. \,\, (\kappa)$', fontsize = fontsize)
        plt.grid(True)
        plt.plot(iteration, diff_his, linewidth=2, label = str(interval))

    plt.axhline(1, linewidth=2, linestyle='--', color='black', label = 'Exact $\kappa_1$,$\kappa_3$')
    plt.axhline(0.05, linewidth=2, linestyle='--', color='black', label = 'Exact $\kappa_2$')
    legend = plt.legend(shadow=True, loc='best', fontsize=18, ncol = 2)
    plt.xticks(np.array([0, 50000, 100000, 150000]))
    plt.tick_params( labelsize = 20)
    fig.set_size_inches(w=11,h=6)
    plt.savefig(''.join([PATH,str(args.case),'_his_coeff','.pdf'])) 


if args.type == 'his_loss':
    for interval in intervals:
        case = args.case + '_'  + str(interval) 
        records = scipy.io.loadmat(''.join([str(case), '_record.mat']))
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
    fig.set_size_inches(w=11,h=6) 
    plt.savefig(''.join([PATH,str(args.case),'_his_loss','.pdf'])) 


if args.type == 'exact':
    delta_test = 0.01 # time step
    T = 1
    xtest = np.linspace(-1,1,500) 
    ttest = np.arange(0, T+delta_test, delta_test)
    if args.case == 'ADE_2bp_balanced':
        fig, axs = plt.subplots(nrows=3,ncols=1,sharex='col', sharey='row')
        (ax3, ax2, ax1) = axs

        ax3.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax3.tick_params( labelsize = 20)
        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize)
        ax1.set_xlabel('$x$', fontsize=fontsize)

        data_temp = []
        for j in range(66, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 1)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[66:101])
        cs3 = ax3.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        data_temp = []
        for j in range(33, 66):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 0.05)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[33:66])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        data_temp = []
        for j in range(0, 33):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 1)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:33])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs3, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)

    if args.case == 'ADE_1bp_imbalanced':
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col', sharey='row', gridspec_kw={'height_ratios': [4, 1]})
        (ax2, ax1) = axs

        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize)
        ax1.set_xlabel('$x$', fontsize=fontsize)

        data_temp = []
        for j in range(20, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], .05)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[20:101])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')


        data_temp = []
        for j in range(0, 20):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 1)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:20])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs2, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)

    if args.case == 'ADE_1bp_balanced':
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col', sharey='row')
        (ax2, ax1) = axs

        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize, loc='bottom')
        ax1.set_xlabel('$x$', fontsize=fontsize)

        data_temp = []
        for j in range(50, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], .05)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[50:101])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        data_temp = []
        for j in range(0, 50):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], 1)] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:50])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs2, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)

    plt.savefig(''.join([PATH,str(args.case),'_exact_solution','.png']), dpi = 400)


if args.type == 'estimated':
    delta_test = 0.01 # time step
    T = 1
    xtest = np.linspace(-1,1,500) 
    ttest = np.arange(0, T+delta_test, delta_test)
    if args.case == 'ADE_2bp_balanced':
        fig, axs = plt.subplots(nrows=3,ncols=1,sharex='col', sharey='row')
        (ax3, ax2, ax1) = axs

        ax3.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax3.tick_params( labelsize = 20)
        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize)
        ax1.set_xlabel('$x$', fontsize=fontsize)

        records = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[68, 101]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(68, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[68:101])
        cs3 = ax3.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        records = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[33, 68]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(33, 68):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[33:68])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        records = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[0, 33]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(0, 33):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:33])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs3, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)

    if args.case == 'ADE_1bp_imbalanced':
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col', sharey='row', gridspec_kw={'height_ratios': [4, 1]})
        (ax2, ax1) = axs

        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize)
        ax1.set_xlabel('$x$', fontsize=fontsize)

        records = scipy.io.loadmat(''.join(['ADE_1bp_imbalanced_[19, 101]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(19, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[19:101])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')

        records = scipy.io.loadmat(''.join(['ADE_1bp_imbalanced_[0, 19]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(0, 19):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:19])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs2, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)
    
    if args.case == 'ADE_1bp_balanced':
        fig, axs = plt.subplots(nrows=2,ncols=1,sharex='col', sharey='row')
        (ax2, ax1) = axs

        ax2.spines['bottom'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.tick_params( labelsize = 20)
        ax1.tick_params( labelsize = 20)
        ax2.set_ylabel('$t$', fontsize=fontsize, loc='bottom')
        ax1.set_xlabel('$x$', fontsize=fontsize)


        records = scipy.io.loadmat(''.join(['ADE_1bp_balanced_[50, 101]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(50, 101):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[50:101])
        cs2 = ax2.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')


        records = scipy.io.loadmat(''.join(['ADE_1bp_balanced_[0, 50]','_record.mat']))
        diff_his  = np.asarray(records['total'])[:,2]
        data_temp = []
        for j in range(0, 50):
            data_temp.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], diff_his[-1])] for i in range(len(xtest))])
        data_temp = np.asarray(data_temp)
        X, Y = np.meshgrid(xtest,ttest[0:50])
        cs1 = ax1.contourf(X, Y, data_temp[:, :, 2], 500, cmap=cm.jet, origin='lower')
        plt.subplots_adjust(hspace = .0)

        cbar = fig.colorbar(cs2, ax = axs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        fig.set_size_inches(w=11, h=11)

    plt.savefig(''.join([PATH,str(args.case),'_estimated_solution','.png']), dpi = 400)

if args.type == 'error':
    delta_test = 0.01 # time step
    T = 1
    xtest = np.linspace(-1,1,500) 
    ttest = np.arange(0, T+delta_test, delta_test)
    if args.case == 'ADE_1bp_balanced':
        records1 = scipy.io.loadmat(''.join(['ADE_1bp_balanced_[0, 50]','_record.mat']))
        records2 = scipy.io.loadmat(''.join(['ADE_1bp_balanced_[50, 101]','_record.mat']))
        diff_his1  = np.asarray(records1['total'])[:,2]
        diff_his2  = np.asarray(records2['total'])[:,2]
        estimated = []
        for j in range(len(ttest)):
            if j < len(ttest) // 2: e = diff_his1[-1]
            else:e = diff_his2[-1]
            estimated.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        estimated = np.asarray(estimated)

        ext = []
        for j in range(len(ttest)):
            if j < len(ttest) // 2: e = 1
            else: e = .05
            ext.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        ext = np.asarray(ext)


        levels = np.linspace(-.007,.007,501)

        fig, ax = plt.subplots()
        X, Y = np.meshgrid(xtest,ttest)
        cs = ax.contourf(X, Y, estimated[:, :, 2]-ext[:, :, 2] , levels = levels, cmap=cm.jet, origin='lower')

        plt.xlabel('$x$', fontsize = fontsize)
        plt.ylabel('$t$', fontsize = fontsize)

        #cbar = fig.colorbar(cs)
        cbar = fig.colorbar(cs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)

        fig.set_size_inches(w=11, h=11)
        ax.tick_params( labelsize = 20)
        
    if args.case == 'ADE_2bp_balanced':
        records1 = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[0, 33]','_record.mat']))
        records2 = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[33, 68]','_record.mat']))
        records3 = scipy.io.loadmat(''.join(['ADE_2bp_balanced_[68, 101]','_record.mat']))
        diff_his1  = np.asarray(records1['total'])[:,2]
        diff_his2  = np.asarray(records2['total'])[:,2]
        diff_his3  = np.asarray(records3['total'])[:,2]

        estimated = []
        for j in range(len(ttest)):
            if j <33: e = diff_his1[-1]
            elif j >33  and j < 68: e = diff_his2[-1]
            else: e = diff_his3[-1]
            estimated.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        estimated = np.asarray(estimated)

        ext = []
        for j in range(len(ttest)):
            if j < len(ttest) // 3: e = 1
            elif j >= len(ttest) // 3  and j < 2*len(ttest) // 3: e = 0.05
            else: e = 1
            ext.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        ext = np.asarray(ext)


        fig, ax = plt.subplots()
        X, Y = np.meshgrid(xtest,ttest)
        cs = ax.contourf(X, Y, estimated[:, :, 2]-ext[:, :, 2], 500, cmap=cm.jet, origin='lower')

        plt.xlabel('$x$', fontsize = fontsize)
        plt.ylabel('$t$', fontsize = fontsize)

        cbar = fig.colorbar(cs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)

        plt.tick_params( labelsize = 20)
        fig.set_size_inches(w=11, h=11)

    if args.case == 'ADE_1bp_imbalanced':
        records1 = scipy.io.loadmat(''.join(['ADE_1bp_imbalanced_[0, 19]','_record.mat']))
        records2 = scipy.io.loadmat(''.join(['ADE_1bp_imbalanced_[19, 101]','_record.mat']))
        diff_his1  = np.asarray(records1['total'])[:,2]
        diff_his2  = np.asarray(records2['total'])[:,2]
        estimated = []
        for j in range(len(ttest)):
            if j < len(ttest) // 5: e = diff_his1[-1]
            else:e = diff_his2[-1]
            estimated.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        estimated = np.asarray(estimated)

        ext = []
        for j in range(len(ttest)):
            if j < len(ttest) // 5: e = 1
            else: e = .05
            ext.append([[xtest[i],ttest[j],u_ext(xtest[i],ttest[j], e)] for i in range(len(xtest))])
        ext = np.asarray(ext)


        levels = np.linspace(-.35,.35,501)

        fig, ax = plt.subplots()
        X, Y = np.meshgrid(xtest,ttest)
        cs = ax.contourf(X, Y, estimated[:, :, 2]-ext[:, :, 2] , levels = levels, cmap=cm.jet, origin='lower')

        plt.xlabel('$x$', fontsize = fontsize)
        plt.ylabel('$t$', fontsize = fontsize)

        cbar = fig.colorbar(cs, shrink=1)
        cbar.ax.tick_params(labelsize = 26)
        ax.tick_params( labelsize = 20)
        fig.set_size_inches(w=11, h=11)
    plt.savefig(''.join([PATH,str(args.case),'_error','.png']), dpi = 400)
