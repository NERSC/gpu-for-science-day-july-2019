# GPU for Science Day Mini-App

Hacking Competition Code for NERSC's GPU for Science Day, July 2019

## Project Layout

- `^/gpp`
  - `gpp.cpp`
    - main function and the kernel to focus on
  - `Makefile`
- `^/external`
  - `commonDefines.h`
  - `arrayMD/`
    - host-device data array
  - `ComplexClass/`
    - custom complex number class
  - `timemory`
    - C++ template library for performance reporting
    - PGI compiler
      - v18.10 will fail to compile this library 
      - v19.5 (default loaded by modules below) will compile this library but not report anything -- this is a compiler bug
  - `cereal`
    - Library used by `timemory` for serialization (not used)

## Development Environment

### Obtaining code

```bash
cd $HOME
git clone https://github.com/NERSC/gpu-for-science-day-july-2019.git
cd gpu-for-science-day-july-2019/gpp
```

### Cori Modules __[REQUIRED]__

```bash
module use /usr/common/software/gpu4sci-jul-2019/modulefiles
```

| Available modules     |
|:----------------------|
| `gpu4science/gcc`     |
| `gpu4science/intel`   |
| `gpu4science/cuda`    |
| `gpu4science/kokkos`  |
| `gpu4science/openmp`  |
| `gpu4science/openacc` |

> **Info: To switch between available modules:**
```bash
make clean
# make sure unload the current module
module unload gpu4science/{gcc,intel,cuda,kokkos,openmp,openacc}
# load the new module
module load gpu4science/{gcc,intel,cuda,kokkos,openmp,openacc}
```

## Cori GPU

### Get GPU node

```bash
salloc -A gpu4sci -C gpu -N 1 -t 04:00:00 -c 10 --gres=gpu:1
```

### Build CPU (sequential) version

```bash
# setup
module load gpu4science/intel

# build
make COMP=intel
```

### CUDA

```bash
# setup
module load gpu4science/cuda

# build
make COMP=cuda
```

### OpenACC

```bash
# setup
module load gpu4science/openacc

# build
make COMP=openacc
```

### OpenMP

```bash
# setup
module load gpu4science/openmp

# build
make COMP=openmp
export OMP_NUM_THREADS=10
```

> **Info: For the Cori GPU Skylake CPU, set OMP_NUM_THREADS=10. (This will not affect the GPU.)**

### Kokkos

```bash
# setup
module load gpu4science/kokkos

# build
make COMP=kokkos
```

## Testing

### Running test/debugging problem

- Fast, good for debugging

```bash
srun ./gpp.ex test
```

> **Hint: Do NOT optimize the test problem, it runs so quickly it is not representative**

### Running benchmark problem

- Slow, this is how we will determine the hackathon winner

```bash
srun ./gpp.ex benchmark
```

> **Hint: Optimize this problem**

## Competition Submission

1. Decide on a team name, if you have not done so already
   - In the steps below, replace `TEAM_NAME` with this name
2. Go to the top-level directory
3. Create a branch for your team
   - `git checkout -b TEAM_NAME`
4. Check to make sure your directory is clean
   - `git status`
   - remove any build files or outputs that show up
   - Do not add/commit any files other than code
5. Stage the files you want to commit
   - `git add gpp/gpp.cpp` and any other relevant **text** files, e.g. `gpp/gpp.cu`
6. Commit the code
   - `git commit -m "Official submission of TEAM_NAME"`
7. Push the code upstream
   - `git push`
8. Execute the PyCTest script
   - `python ./pyctest-runner.py --team=TEAM_NAME --compiler=COMPILER`
     - `TEAM_NAME` should be your team name... Please do not copy/paste and submit as `TEAM_NAME`
     - `COMPILER` should be one of `openacc`, `openmp`, `cuda`, or `kokkos`
   - This script will:
     - Build the code in the `gpp` folder
     - Execute the benchmark test
     - Submit the build and test logs to [cdash.nersc.gov](https://cdash.nersc.gov/index.php?project=gpu-for-science-day-july-2019)
