source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module add compilers/gcc/9.3.0
#module add libs/cudnn/8.2.1_cuda11.3
module add libs/cudnn/8.4.0.27_cuda11.x
#module add compilers/cuda/11.3.0
module add compilers/cuda/11.7.0
module add libs/openblas/0.3.25_gcc9.3.0