import matplotlib.pyplot as plt
from helper import latexify
import numpy as np
import re
import pickle
from scipy.stats import norm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

def load_chain(filepath, shape_variables):
    rms_chain = []
    # shape_variables = np.zeros([4, 6, 1])
    id_chain = []
    nm = 0
    f = open(filepath, 'rb')
    while 1:
        try:
            state = pickle.load(f)
            rms_chain.append(state['rms'])
            shape_variables[:, :, nm] = state['dfn']
            id_chain.append(state['dfn_id'])
            nm += 1
        except EOFError:
            break
    return rms_chain, shape_variables, id_chain

def variable_hist(var_idx, shape_vars, save_flag=True, **kwargs):

    var = shape_vars[var_idx[0], var_idx[1], :]
    bin = kwargs.get('bins', np.arange(var.min(), var.max(), 0.05))
    (mu, sigma) = norm.fit(var)

    latexify(fig_width=6)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(var,bins=bin, density=True,
                                edgecolor='black', linewidth=1.2, histtype='bar')

    y = mlab.normpdf(bins, mu, sigma)
    l = ax1.plot(bins, y, linewidth=2)

    return ax1

if __name__ == '__main__':

    project_path = '/Users/shiyili/euler_remote/scratch/Case3_1x1x1_cxyz_2/inverse'
    # project_path = '/Users/shiyili/euler_remote/home/model_1x1x1_cxyz_1000/inverse'

    shape_var_size = np.zeros([4, 6, 1001])

    rms_log, shape_var, model_ids = load_chain(project_path+'/mcmc_chain.pkl', shape_var_size)

    del shape_var_size

    ax_hist_cx = variable_hist([3, 0], shape_var, bins=np.arange(-0.5, 0.5, 0.02))
    ax_hist_cx.set_xlabel('X-coordinate of fracture center')
    ax_hist_cx.set_ylabel('Density')
    ax_hist_cx.set_title('X-coordinate of fracture center')
    plt.xticks(np.arange(-0.5, 0.5, 0.1))
    plt.savefig(project_path + '/mcmc_cluster_cx.pdf')
    plt.show()
    #
    ax_hist_cy = variable_hist([3, 1], shape_var, bins=np.arange(-0.5, 0.5, 0.02))
    ax_hist_cy.set_xlabel('Y-coordinate of fracture center')
    ax_hist_cy.set_ylabel('Density')
    ax_hist_cy.set_title('Y-coordinate of fracture center')
    plt.xticks(np.arange(-0.5, 0.5, 0.1))
    plt.savefig(project_path + '/mcmc_cluster_cy.pdf')
    plt.show()

    ax_hist_cz = variable_hist([3, 2], shape_var, bins=np.arange(-0.5, 0.5, 0.02))
    ax_hist_cz.set_xlabel('Z-coordinate of fracture center')
    ax_hist_cz.set_ylabel('Density')
    ax_hist_cz.set_title('Z-coordinate of fracture center')
    plt.xticks(np.arange(-0.5, 0.5, 0.1))
    plt.savefig(project_path + '/mcmc_cluster_cz.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shape_var[3, 0, :], shape_var[3, 1, :], shape_var[3, 2, :])
    ax.set_xlabel('cx')
    ax.set_ylabel('cy')
    ax.set_zlabel('cz')
    ax.set_title('Spatial Distribution of the inferred fracture center')
    plt.savefig(project_path + '/center_distribution.pdf')
    plt.show()
    #
    fig = plt.figure()
    plt.plot(rms_log)
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Evolution as MCMC Iteration')
    plt.savefig(project_path + '/rms_iteration.pdf')
    plt.show()

    print(len(rms_log))
    print(rms_log[0:5])
    #
    # Analyze result
    # with open(project_path + '/mcmc_log.txt', 'r') as logfile:
    #     mcmc_log = logfile.readlines()
    #     separator = '*' * 60 + '\n'
    #     idx_sep = [i for i, j in enumerate(mcmc_log) if j == separator]
    #
    #     n_fractures = idx_sep[1] - idx_sep[0] - 3
    #     n_model = len(idx_sep)
    #
    #     rms = []
    #     for i in idx_sep:
    #         rms.append(float(mcmc_log[i + 2].replace('\n', '')))
    #
    #     shape_variables = np.zeros([n_fractures, 6, n_model])
    #
    #     nm = 0
    #
    #     p = re.compile(r'\d+\.\d*|-\d+\.\d*')
    #
    #     for i in idx_sep:
    #         nf = 0
    #         var_list = mcmc_log[i+3: i+n_fractures+3]
    #         print(i)
    #         for var_str in var_list:
    #             digi = p.findall(var_str)
    #             var_ary = np.asarray(list(map(float, digi)))
    #             print(var_ary)
    #             shape_variables[nf, :, nm] = var_ary
    #             nf += 1
    #         nm += 1

    # #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.hist(cy, bins=np.arange(-1, 1, 0.05), density=True,
    #          edgecolor='black', linewidth=1.2, histtype='bar')
    # ax1.set_xlabel('y-coordinate of fracture center')
    # ax1.set_ylabel('Density')
    # ax1.set_title('y-coordinate of fracture center')
    # # plt.xticks(np.arange(-0.5, 0.5, 0.1))
    # plt.savefig(project_path + '/mcmc_cluster_cy.pdf')
    # plt.show()
    # #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.hist(cz, bins=np.arange(-1, 1, 0.05), density=True,
    #          edgecolor='black', linewidth=1.2, histtype='bar')
    # ax1.set_xlabel('z-coordinate of fracture center')
    # ax1.set_ylabel('Density')
    # ax1.set_title('z-coordinate of fracture center')
    # # plt.xticks(np.arange(-0.5, 0.5, 0.1))
    # plt.savefig(project_path + '/mcmc_cluster_cz.pdf')
    # plt.show()
