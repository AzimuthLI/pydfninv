import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import re, pickle
from IPython.display import set_matplotlib_formats, Image

def nbplotstyle(style='seaborn-pastel', figsize=[9, 6]):
    set_matplotlib_formats('retina')
    mpl.style.use(style)
    params = {'backend': 'ps',
              'axes.labelsize': 12,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 14,
              'axes.labelweight': 'bold',
              'font.size': 12,  # was 10
              'legend.fontsize': 12,  # was 10
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize': figsize,
              'font.family': 'Sans-serif'
              }
    mpl.rcParams.update(params)

def load_chain(filepath):
    rms_chain = []
    id_chain = []
    shape_vars = []
    f = open(filepath, 'rb')
    while True:
        try:
            state = pickle.load(f)
            rms_chain.append(state['rms'])
            shape_vars.append(state['dfn'])
            id_chain.append(state['dfn_id'])
        except EOFError:
            break
    f.close()

    return rms_chain, shape_vars, id_chain

def parse_jobreport(job_report):
    report = open(job_report, 'r').readlines()
    separator = '== RICHARDS FLOW {}\n'.format('='*63)
    idx_sep = [i for i, j in enumerate(report) if j == separator]
    idx_sep.append(idx_sep[-1]+12)

    time_str = []
    writevtk = []
    boundary_label = []
    pattern_time = re.compile(r' Step(.*)Time=(.*)Dt=.*')
    pattern_writevtk = re.compile(r' --> write rate output file: dfn_explicit_(.*)-darcyvel-(.*)')
    for i in range(len(idx_sep)-1):
        vtk_flag = []
        for line in report[idx_sep[i]:idx_sep[i+1]]:
            matchObj_time = pattern_time.match(line)
            matchObj_writevtk = pattern_writevtk.match(line)
            if matchObj_time:
                time_str.append(matchObj_time.group(2))
            if matchObj_writevtk:
                vtk_flag.append(True)
                bc = matchObj_writevtk.group(1)
            else:
                vtk_flag.append(False)
        if any(vtk_flag):
            writevtk.append(True)
            boundary_label.append(bc)
        else:
            writevtk.append(False)
    time = np.asarray([float(e) for e in time_str], dtype=np.float64)

    time = time[writevtk]
    d = {}
    for bc in set(boundary_label):
        d.update({bc: time[[i for i, j in enumerate(boundary_label) if j == bc]]})

    return d

def plot_synthetic(time:np.array, data:pd.DataFrame):
    fig = plt.figure(figsize=[6, 9])
    boundaries = ['front_back', 'left_right', 'top_bottom']
    cols = syn_data.columns
    i = 0
    j = 1
    for bc in boundaries:
        if bc in list(time.keys()):
            ax = fig.add_subplot(len(time.keys()), 1, j)
            for col in cols:
                ax.plot(time[bc], syn_data.iloc[i:len(time[bc])*j][col], '*-', label='Station ' + col.split('_')[-1])
            ax.set_xscale('log')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Pressure(MPa)')
            ax.set_title('Synthethic Observation on ' + bc, fontstyle='italic')
            lgd = ax.legend(loc=7, bbox_to_anchor=(1.25, 0.5))
            # fig.savefig(project_dir + '/inverse/synthetic_obs.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

            i += len(time[bc])
            j += 1
    plt.show()
    return None


if __name__ == '__main__':

    nbplotstyle(style='seaborn-darkgrid')

    scratch_dir = '/Users/shiyili/euler_remote/scratch'
    # cases = [s for s in os.listdir(scratch_dir) if 'Case' in s]
    #
    # case_id = 5
    # project_dir = scratch_dir + '/' + cases[case_id]
    project_dir = scratch_dir + '/' + 'dfn3_size1_xyz_dr0'
    print('Working with {}'.format(project_dir))

    # Synthetic Observation Data
    syn_observation = project_dir + '/synthetic/forward_simulation/forward_results.csv'
    syn_jobreport = project_dir + '/synthetic/job_report.txt'
    syn_data = pd.read_csv(syn_observation, index_col=0)
    time = parse_jobreport(syn_jobreport)

    plot_synthetic(time, syn_data)
    #
    # fig = plt.figure(figsize=[9, 6])
    # ax = fig.add_subplot(111)
    # cols = syn_data.columns
    # for col in cols:
    #     ax.plot(time, syn_data[col], '*-', label='Station ' + col.split('_')[-1])
    # ax.set_xscale('log')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Pressure(MPa)')
    # ax.set_title('Synthethic Observation', fontstyle='italic')
    # lgd = ax.legend(loc=7, bbox_to_anchor=(1.25, 0.5))
    # fig.savefig(project_dir + '/inverse/synthetic_obs.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

    # Load inverse results
    pkl_file = project_dir + '/inverse/mcmc_chain.pkl'
    rms_log, shape_log, id_log = load_chain(pkl_file)

    # # Root-Mean-Square Error Log of MCMC
    fig = plt.figure()
    plt.plot(rms_log, color='orangered')
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.title('Root-Mean-Square Error Log of MCMC', fontstyle='italic')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/rms_iteration.pdf')
    plt.show()
    #
    # Probabalistic distribution of variables -- cx
    cx = np.asarray([e[3, 0] for e in shape_log])
    print(cx)
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-1, 1, 0.1)
    sns.distplot(cx, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('X-coordinate of the fracture center', fontstyle='italic')
    ax.set_xticks(np.arange(-1, 1, 0.1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cx_cluster.pdf')
    plt.show()

    # Probabalistic distribution of variables -- cy
    cy = np.asarray([e[3, 1] for e in shape_log])
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-0.5, 0.5, 0.1)
    sns.distplot(cy, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('Y-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('Y-coordinate of the fracture center', fontstyle='italic')
    ax.legend()
    ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cy_cluster.pdf')
    plt.show()
    #
    # Probabalistic distribution of variables -- cz
    cz = np.asarray([e[3, 2] for e in shape_log])
    f, ax = plt.subplots(1, 1)
    bins = np.arange(-0.5, 0.5, 0.1)
    sns.distplot(cz, bins=bins, ax=ax, fit=norm,
                 kde_kws={"color": "r", "lw": 0.5, "label": "KDE"},
                 fit_kws={"color": "g", "lw": 1.5, "label": "Normal"})
    ax.set_xlabel('Z-coordinate')
    ax.set_ylabel('Sample Density')
    ax.set_title('Z-coordinate of the fracture center', fontstyle='italic')
    ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_dir + '/inverse/cz_cluster.pdf')
    plt.show()