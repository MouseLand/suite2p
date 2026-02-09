import os

# run caiman and fiola

for i in range(3):
    bsub = f'bsub -n 16 -R"select[sapphirerapids]" -o outcm{i}.out \
            "source ~/.bashrc; source ~/add_mini.sh; source activate cm; \
             ~/miniforge3/envs/cm/bin/python /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_caiman.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --roi {i} --n_processes 32  > outcm{i}.log"'
    print(bsub)
    os.system(bsub)
    
for i in range(3):
    bsub = f'bsub -n 12 -gpu "num=1" -q gpu_a100  -o outtf{i}.out \
            "source ~/.bashrc; source ~/add_mini.sh; source activate tf; \
             python /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_fiola.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --roi {i}  > outtf{i}.log"'
    print(bsub)
    os.system(bsub)

### timings

nfr = [500, 1000, 2000, 4000, 8000, 16000, 32000]

# suite2p nonrigid
for i in nfr:
    bsub = f"bsub -n 12 -gpu 'num=1' -q gpu_a100 -o out{i}.out\
        'source ~/add_mini.sh; source activate s2p; python \
            /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_suite2p.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --tfr {i} > out{i}.log'"
    print(bsub)
    os.system(bsub)

# suite2p rigid
for i in nfr:
    bsub = f"bsub -n 12 -gpu 'num=1' -q gpu_a100 -o outr{i}.out \
        'source ~/add_mini.sh; source activate s2p; python \
            /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_suite2p.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --rigid --tfr {i} > outr{i}.log'" 
    print(bsub)
    os.system(bsub)

# fiola
for i in nfr:
    bsub = f"bsub -n 12 -gpu 'num=1' -q gpu_a100  -o outr{i}_fiola.out \
        'source ~/.bashrc; source ~/add_mini.sh; source activate tf; \
          python /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_fiola.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --timing --tfr {i} > outr{i}_fiola.log'"
    print(bsub)
    os.system(bsub)

nproc = 16

# caiman nonrigid
for i in nfr:
    bsub = f"bsub -n {nproc} -R'select[sapphirerapids]' -o out_{nproc}_{i}.out \
        'source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python \
            /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_caiman.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --timing --n_processes {nproc} --tfr {i} > out{i}_{nproc}.log'"
    print(bsub)
    os.system(bsub)

# caiman rigid
for i in nfr:
    bsub = f"bsub -n {nproc} -R'select[sapphirerapids]' -o outr_{nproc}_{i}.out \
        'source ~/add_mini.sh; source activate cm; ~/miniforge3/envs/cm/bin/python \
            /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/reg_caiman.py \
            --root /groups/stringer/stringerlab/suite2p_paper/reg_benchmarks/GT1/ --timing --n_processes {nproc} --tfr {i} --rigid > outr{i}_{nproc}.log'"
    print(bsub)
    os.system(bsub)

