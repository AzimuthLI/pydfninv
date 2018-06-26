import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from nbhelper import nbplotstyle, parse_jobreport, load_chain
import seaborn as sns
import os, re


if __name__ == '__main__':

    nbplotstyle(style='seaborn-darkgrid')

    scratch_dir = '/Users/shiyili/euler_remote/scratch'

    cases = [s for s in os.listdir(scratch_dir) if 'Case' in s]

    project_dir = scratch_dir + '/' + cases[7]

    # Synthetic Observation Data
    syn_observation = project_dir + '/synthetic/forward_simulation/forward_results.csv'
    syn_jobreport = project_dir + '/synthetic/job_report.txt'
    syn_data = pd.read_csv(syn_observation, index_col=0)
    time = parse_jobreport(syn_jobreport)

    fig = plt.figure(figsize=[9, 6])
    cols = syn_data.columns
    for col in cols:
        plt.plot(time, syn_data[col], '*-', label='Station ' + col.split('_')[-1])
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure(MPa)')
    plt.title('Synthethic Observation', fontstyle='italic')
    plt.legend(loc=7, bbox_to_anchor=(1.25, 0.5))
    plt.savefig(project_dir + '/inverse/rms_iteration.pdf')
    plt.show()

    # Load inverse results
    pkl_file = project_dir + '/inverse/mcmc_chain.pkl'
    rms_log, shape_log, id_log = load_chain(pkl_file)

    # Root-Mean-Square Error Log of MCMC
    fig = plt.figure()
    plt.plot(rms_log, color='orangered')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Log of MCMC', fontstyle='italic')
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/rms_iteration.pdf')
    plt.show()

    # Probabalistic distribution of variables -- cx
    cx = np.asarray([e[3, 0] for e in shape_log])
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-0.5, 0.5, 0.05)
    sns.distplot(cx, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('X-coordinate of the fracture center', fontstyle='italic')
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cx_cluster.pdf')
    plt.show()

    # Probabalistic distribution of variables -- cx
    cy = np.asarray([e[3, 1] for e in shape_log])
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-0.5, 0.5, 0.05)
    sns.distplot(cy, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('Y-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('Y-coordinate of the fracture center', fontstyle='italic')
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cy_cluster.pdf')
    plt.show()

    # Probabalistic distribution of variables -- cx
    cz = np.asarray([e[3, 2] for e in shape_log])
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-0.5, 0.5, 0.05)
    sns.distplot(cz, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('Z-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('Z-coordinate of the fracture center', fontstyle='italic')
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cz_cluster.pdf')
    plt.show()