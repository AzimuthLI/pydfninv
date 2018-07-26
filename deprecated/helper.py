"""
.. module:: helper_2.py
   :synopsis: helper methods
.. moduleauthor:: Shiyi Li

"""
import os, shutil, sys
from string import Template
from time import time, sleep

def write_template_file(self, src, dst, para_list):

    template_file = open(src, 'r').read()
    generate_file = open(dst, 'w+')
    s = Template(template_file)
    generate_file.write(s.safe_substitute(para_list))
    generate_file.close()

def write_forward_simulation_options(self, input_para_lists):

    input_files = self._project + '/dfnWorks_input_files/'
    template_files = os.environ['PYDFNINV_PATH'] + 'dfnWorks_input_templates/'

    if input_para_lists['dfnGen'] == None:
        shutil.copy2(template_files+'gen_user_ellipses.dat', input_files+'gen_user_ellipses.dat')
    else:
        self.write_template_file(template_files+'gen_user_ellipses.i',
                                 input_files + 'gen_user_ellipses.dat',
                                 input_para_lists['dfnGen'])

    if input_para_lists['dfnFlow'] == None:
        shutil.copy2(template_files+'dfn_explicit.in', input_files+'dfn_explicit.in')
    else:
        self.write_template_file(template_files+'dfn_explicit.i',
                                 input_files + 'dfn_explicit.in',
                                 input_para_lists['dfnFlow'])

    if input_para_lists['dfnTrans'] == None:
        shutil.copy2(template_files+'PTDFN_control.dat', input_files+'PTDFN_control.dat')
    else:
        self.write_template_file(template_files+'PTDFN_control.i',
                                 input_files + 'PTDFN_control.dat',
                                 input_para_lists['dfnTrans'])

    if input_para_lists['user_define'] == None:
        shutil.copy2(template_files + 'define_user_ellipses.dat', input_files + 'define_user_ellipses.dat')
    else:
        self.write_template_file(template_files + 'define_user_ellipses.i',
                                 input_files + 'define_user_ellipses.dat',
                                 input_para_lists['user_define'])

    dfn_inputs = {'dfnGen': input_files + '/gen_user_ellipses.dat',
                  'dfnFlow': input_files + '/dfn_explicit.in',
                  'dfnTrans': input_files + '/PTDFN_control.dat',
                  }

    input_file = open(self._forward_model + 'input_dfnWorks.txt', 'w')
    input_file.write('dfnGen ' + dfn_inputs['dfnGen'] + '\n' +
                     'dfnFlow ' + dfn_inputs['dfnFlow'] + '\n' +
                     'dfnTrans ' + dfn_inputs['dfnTrans'])
    input_file.close()

    options = {'jobname': self._forward_model,
               'ncpu': self._ncpu,
               'input_file': self._forward_model + 'input_dfnWorks.txt',
               'cell': False
               }
    return options

def make_direcories(self, field_obs):

    if os.path.isdir(self._project):
        if not os.listdir(self._project) == []:
            for root, dirs, files in os.walk(self._project):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
    else:
        os.mkdir(self._project)

    if not os.path.isdir(self._accepted_models):
        os.mkdir(self._accepted_models)
    if not os.path.isdir(self._forward_model):
        os.mkdir(self._forward_model)
    if not os.path.isdir(self._data_plots):
        os.mkdir(self._data_plots)

    if field_obs == ' ':
        if not os.path.exists(self._field_obs):
            raise Exception("Error: No Field Observation is given")
    else:
        if os.path.dirname(field_obs) != self._project:
            shutil.copyfile(field_obs, self._field_obs)
        else:
            os.rename(field_obs, self._field_obs)

    os.mkdir(self._project+'/dfnWorks_input_files')


def update_progress(i, total, tic):
    progress = i / total
    toc = time()
    dt = toc - tic

    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}, {3} / {4} finished. Average time: {5:.5f}" \
        .format("#" * block + "-" * (barLength - block), progress * 100, status, i, total, dt / i)
    sys.stdout.write(text)
    sys.stdout.flush()
