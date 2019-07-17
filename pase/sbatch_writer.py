import os

class submission_writer(object):

    def __init__(self, job_name, out_dir, memory, asr_pth, skp_pth, emo_pth, lang_pth):

        self.job_name = job_name
        self.out_dir = out_dir
        self.memory = memory
        self.tasks = {'ASR' : asr_pth, 'spk_id' : skp_pth, 'EMO' : emo_pth, 'LANG' : lang_pth}

    def write(self, sbatch_file_name, cmd):
        out_dir = "./downstream_submissions/"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        write_slurm_submission_file(os.path.join(out_dir, sbatch_file_name),
                                    self.job_name,
                                    self.out_dir,
                                    self.memory,
                                    cmd)

    def cmd_maker(self, pase_cfg, latest_ckpt, data_root, res_pth):
        cmds = []
        for name, run_file in self.tasks.items():
            cmd = "python {} {} {} {} {}\n".format(run_file, pase_cfg, latest_ckpt, data_root, res_pth + name)
            cmds.append(cmd)
        return cmds

    def __call__(self, sbatch_file_name, pase_cfg, latest_ckpt, data_root, res_pth):
        cmd = self.cmd_maker(pase_cfg, latest_ckpt, data_root, res_pth)
        self.write(sbatch_file_name, cmd)



def write_slurm_submission_file(sbatch_file_name, job_name, out_dir, memory, run_command_lines, **kwargs):
    """Create a Slurm job submission file based on resource requirements and the set of commands that need to be run.
    :param sbatch_file_name:
    :param job_name:
    :param walltime:
    :param memory:
    :param run_command_lines:
    :param processors:
    :param partition:
    :return:
    """
    writer = open(sbatch_file_name, "w")
    writer.write("#!/bin/bash\n\n")
    writer.write("#SBATCH --job-name=" + job_name + "\n")
    writer.write("#SBATCH --nodes=1" + "\n")
    writer.write("#SBATCH --cpus-per-task=8" + "\n")
    writer.write("#SBATCH --mem=" + str(memory) + "\n")
    writer.write("#SBATCH --output={}\n".format(os.path.join(out_dir, "{}.%j.out".format(job_name))))
    writer.write("#SBATCH -t 5-00:00:00 \n")

    if kwargs is not None:
        for key, arg in kwargs.items():
            writer.write("#SBATCH --{0:s}={1:s}\n".format(key, arg))


    writer.write("\n")
    writer.writelines(run_command_lines)
    writer.close()


def read_slurm_submission_file(sbatch_file_name):
    """Read the Slurm scheduler parameters and run commands from a Slurm job submission file.
    :param sbatch_file_name:
    :return:
    """
    reader = open(sbatch_file_name)
    sbatch_lines = reader.readlines()
    reader.close()
    slurm_parameters = {}
    for line in [l.strip("\n") for l in sbatch_lines if "SBATCH" in l]:
        splitline = line.split("--")[1].split("=")
        slurm_parameters[splitline[0]] = splitline[1]
    run_command_lines = [l for l in sbatch_lines if ("#" not in l and len(l) > 1)]
    return (slurm_parameters, run_command_lines)



if __name__ == "__main__":
    swriter = submission_writer("test", "../log", memory=32,
                               asr_pth="/ASR/run_TIMIT_fast.py",
                               skp_pth="/spk_id/run_minivox_fast.py",
                               emo_pth="/emorec/run/iemocap_fast.py",
                               lang_pth="/??"
                               )

    swriter("test_writer", "cfg", "ckpt", "data_root", "res_path")
