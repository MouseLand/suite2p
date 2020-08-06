import sys, os, time, glob
from pathlib import Path
from natsort import natsorted
import paramiko
import numpy as np

def unix_path(path):
    return str(path).replace(os.sep, '/')

def ssh_connect(host, username, password):
    """ from paramiko example """
    i=0
    while True:
        print("Trying to connect to %s (attempt %i/30)" % (host, i+1))
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=username, password=password)
            print("Connected to %s" % host)
            break
        except paramiko.AuthenticationException:
            print("Authentication failed when connecting to %s" % host)
            sys.exit(1)
        except:
            print("Could not SSH to %s, waiting for it to start" % host)
            i += 1
            time.sleep(2)
        # If we could not connect within time limit
        if i == 30:
            print("Could not connect to %s. Giving up" % host)
            sys.exit(1)
    return ssh

def send_jobs(save_folder,
         host=None, 
         username=None, 
         password=None, 
         server_root=None, 
         n_cores=4):

    """ send each plane to compute on server separately

    add your own host, username, password and path on server 
    for where to save the data

    """

    if host is None:
        raise Exception('No server specified, please edit suite2p/io/server.py')
    
    # server_root is different from where you created the binaries
    # this line will depend on your set up
    save_folder_server =  Path(*Path(save_folder).parts[1:])
    save_folder_server = Path(server_root) / save_folder_server
    save_path0_server = Path(*Path(save_folder_server).parts[:-1])
    save_folder_name = Path(save_folder).parts[-1]
    print('save path on server: ', unix_path(save_path0_server))
    ssh = ssh_connect(host, username, password)    

    # create bash file in home directory to run
    run_script = Path.home().joinpath('.suite2p/run_script.sh')
    if run_script.exists():
        os.remove(run_script)
    with open(run_script, 'x', newline='') as f:
        f.write('#!/bin/bash\n')    
        # server specific commands to activate python
        f.write('source ~/add_anaconda.sh\n')
        f.write('eval $(~/anaconda4/bin/conda shell.bash hook)\n')
        # activate suite2p environment 
        f.write('source activate suite2p\n')
        # run suite2p single plane command with ops as argument
        f.write('python -m suite2p --single_plane --ops "$@"')

    ssh.exec_command('rm ~/run_script.sh')
    ssh.exec_command('chmod 777 ~/')
    ftp_client=ssh.open_sftp()
    ftp_client.put(run_script, 'run_script.sh')
    ssh.exec_command('chmod 777 run_script.sh')

    pdirs = natsorted(glob.glob(save_folder + '/*/'))
    for k, pdir in enumerate(pdirs):
        ipl = int(Path(pdir).parts[-1][5:])
        print('>>>>>>>>>> PLANE %d <<<<<<<<<'%ipl)
        ops_path_orig = pdir + 'ops.npy'
        op = np.load(ops_path_orig, allow_pickle=True).item()
        fast_disk_orig = Path(op['fast_disk'])

        ## change paths
        op['save_path0'] = unix_path(save_path0_server)
        op['save_folder'] = save_folder_name
        save_path = save_path0_server / save_folder_name / ('plane%d'%ipl)
        op['save_path'] = unix_path(save_path)
        op['fast_disk'] = unix_path(save_path)
        op['ops_path'] = unix_path(save_path / 'ops.npy')
        print(op['ops_path'])
        ## move binary files to server if needed
        # check if file structure needs to be created on remote server
        copy = False
        try:
            ftp_client.stat(op['save_path'])
        except IOError:
            print('copying files')
            ftp_client.mkdir(op['save_path'])
            copy = True
        op['reg_file'] = unix_path(save_path / 'data.bin')
        if 'raw_file' in op:
            op['raw_file'] = unix_path(save_path / 'data_raw.bin')
            if copy:
                ftp_client.put(fast_disk_orig / 'data_raw.bin', op['raw_file'])
            if 'raw_file_chan2' in op:
                op['raw_file_chan2'] = unix_path(save_path / 'data_chan2_raw.bin')
                if copy:
                    ftp_client.put(fast_disk_orig / 'data_raw_chan2.bin', op['raw_file_chan2'])
        else:
            if copy:
                ftp_client.put(fast_disk_orig / 'data.bin', op['reg_file'])
            if 'reg_file_chan2' in op:
                op['reg_file_chan2'] = unix_path(save_path / 'data_chan2.bin')
                if copy:
                    ftp_client.put(fast_disk_orig / 'data_chan2.bin', op['reg_file_chan2'])
        
        # save final version of ops and send to server
        np.save(ops_path_orig, op)
        ftp_client.put(ops_path_orig, op['ops_path'])

        # run plane (server-specific command)
        run_command = '''bsub -n %d -J test_s2p%d -R"select[avx512]" -o out%d.txt "~/run_script.sh '%s' > log%d.txt"'''%(n_cores, ipl, ipl, op['ops_path'], ipl)
        stdin, stdout, stderr = ssh.exec_command(run_command)
        print(stdout.readlines()[0])
    
    ftp_client.close()
    
    print("Command done, closing SSH connection")
    ssh.close()

if __name__ == '__main__':
    send_jobs()