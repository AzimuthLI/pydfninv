import matplotlib.pyplot as plt
from helper import latexify
import numpy as np
import re
import pickle
from scipy.stats import norm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D



if __name__ == '__main__':

    project_path = '/Volumes/SD_Card/Thesis_project/model_2'

    with open(project_path+'/mcmc_chain.pkl', 'rb') as f:
        chain = pickle.load(f)

    n_fractures = chain[0]['dfn'].shape[0]
    n_model = len(chain)

    rms_log = []
    shape_variables = np.zeros([n_fractures, 6, n_model])
    nm = 0

    for model in chain:
        rms_log.append(model['rms'])
        shape_variables[:, :, nm] = model['dfn']
        nm += 1

    print(rms_log)

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

    print(n_model)
    cx = shape_variables[3, 0, :]
    cy = shape_variables[3, 1, :]
    cz = shape_variables[3, 2, :]

    (mu, sigma) = norm.fit(cx)
    print(mu, sigma)
    # # print(cx)
    latexify(fig_width=6)
    #
    # for c in [cx, cy, cz]:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n, bins, patches = ax1.hist(cx,
                                bins=np.arange(-0.3, 0.3, 0.02), density=True,
                                edgecolor='black', linewidth=1.2, histtype='bar')

    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, linewidth=2)

    ax1.set_xlabel('x-coordinate of fracture center')
    ax1.set_ylabel('Density')
    ax1.set_title('x-coordinate of fracture center')
    plt.xticks(np.arange(-0.3, 0.3, 0.1))
    plt.savefig(project_path + '/mcmc_cluster_cx.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cx, cy, cz)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


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

    fig = plt.figure()
    plt.plot(rms_log)
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Evolution as MCMC Iteration')
    plt.savefig(project_path + '/rms_iteration.pdf')
    plt.show()


