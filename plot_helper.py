import matplotlib.pyplot as plt
from helper import latexify
import numpy as np
import re

if __name__ == '__main__':

    project_path = '/Volumes/SD_Card/Thesis_project/model_2'

    # Analyze result
    with open(project_path + '/mcmc_log.txt', 'r') as logfile:
        mcmc_log = logfile.readlines()
        separator = '*' * 60 + '\n'
        idx_sep = [i for i, j in enumerate(mcmc_log) if j == separator]

        n_fractures = idx_sep[1] - idx_sep[0] - 3
        n_model = len(idx_sep)

        rms = []
        for i in idx_sep:
            rms.append(float(mcmc_log[i + 2].replace('\n', '')))

        shape_variables = np.zeros([n_fractures, 6, n_model])

        nm = 0

        p = re.compile(r'\d+\.\d*|-\d+\.\d*')

    #     for i in idx_sep:
    #         nf = 0
    #         var_list = mcmc_log[i+3: i+n_fractures+3]
    #         for var_str in var_list:
    #             digi = p.findall(var_str)
    #             var_ary = np.asarray(list(map(float, digi)))
    #             shape_variables[nf, :, nm] = var_ary
    #             nf += 1
    #         nm += 1
    # print(n_model)
    # cx = shape_variables[3, 0, :]
    # # print(cx)
    # latexify(fig_width=6)
    #
    # fig = plt.figure()
    # plt.hist(cx, bins=np.arange(-0.5, 0.5, 0.05), density=True, edgecolor='black', linewidth=1.2, histtype='bar')
    # plt.xlabel('x-coordinate of fracture center')
    # plt.ylabel('Density')
    # plt.title('Clusters of MCMC samples for x-coordinate of fracture center')
    # plt.xticks(np.arange(-0.5, 0.5, 0.1))
    # plt.savefig(project_path + '/mcmc_cluster_cx.pdf')
    # plt.show()




    fig = plt.figure()
    plt.plot(rms)
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Evolution as MCMC Iteration')
    plt.savefig(project_path + '/rms_iteration.pdf')
    plt.show()


