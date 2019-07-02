#!/bin/bash -e

: ${DEFAULT_ARGS:="--type=benchmark --team=nersc"}

reset-modules()
{
    module purge
    module list
}

echo -e "\n\n\n####### intel #######\n"
reset-modules
module load gpu4science/intel
python ./pyctest-runner.py --compiler=intel -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### gcc #######\n"
reset-modules
module load gpu4science/gcc
python ./pyctest-runner.py --compiler=gcc -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### cuda #######\n"
reset-modules
module load gpu4science/cuda
python ./pyctest-runner.py --compiler=cuda -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### openacc #######\n"
reset-modules
module load gpu4science/openacc
python ./pyctest-runner.py --compiler=openacc -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### openmp #######\n"
reset-modules
module load gpu4science/openmp
python ./pyctest-runner.py --compiler=openmp -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### kokkos #######\n"
reset-modules
module load gpu4science/kokkos
python ./pyctest-runner.py --compiler=kokkos -d gpp ${DEFAULT_ARGS}

echo -e "\n\n\n####### Testing $PWD #######\n"
reset-modules
module load gpu4science/cuda
python ./pyctest-runner.py --compiler=cuda -d solution/cuda ${DEFAULT_ARGS}

echo -e "\n\n\n####### Testing $PWD #######\n"
reset-modules
module load gpu4science/openacc
python ./pyctest-runner.py --compiler=openacc -d solution/openacc ${DEFAULT_ARGS}

echo -e "\n\n\n####### Testing $PWD #######\n"
reset-modules
module load gpu4science/openmp
python ./pyctest-runner.py --compiler=openmp -d solution/openmp ${DEFAULT_ARGS}

echo -e "\n\n\n####### Testing $PWD #######\n"
reset-modules
module load gpu4science/openacc
python ./pyctest-runner.py --compiler=openacc -d solution-rookie/openacc ${DEFAULT_ARGS}

echo -e "\n\n\n####### Testing $PWD #######\n"
reset-modules
module load gpu4science/openmp
python ./pyctest-runner.py --compiler=openmp -d solution-rookie/openmp ${DEFAULT_ARGS}
