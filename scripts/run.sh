###加载环境
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
source /home/share/huadjyin/home/s_sukui/envs/env_cuda17

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate deepgo2
echo 'Conda environment activated: deepgo2'

##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

##Config nnodes node_rank master_addr
NNODES=$1
GPUS=$2
HOSTFILE=$3
HOST=`hostname`
flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
echo $MASTER_IP

HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_IP \
    --master_port 30349
"
echo $DISTRIBUTED_ARGS

echo "
run.sh ----单机单卡RUN-----------
NNODES=${NNODES},
HOST=${HOST},
HOSTFILE=${HOSTFILE},
MASTER_IP=${MASTER_IP},
HOST_RANK=${HOST_RANK},
NODE_RANK=${NODE_RANK}
---------------------------"

##Start torchrun
# nvidia-smi
arr=("cc" "bp" "mf")
data_dir="./data"
epoch=20
device="cuda"

#model="dgg"
#for i in "${arr[@]}"; do
#    echo ---------------ontology $i----------------
#    echo train_dgg.py -m $model -ont $i -dr $data_dir -ep $epoch -d $device
#    python train_dgg.py -m $model -ont $i -dr $data_dir -ep $epoch -d $device
#    echo evaluate.py -m $model -ont $i -dr $data_dir
#    python evaluate.py -m $model -ont $i -dr $data_dir
#done


model="deepgozero_esm_plus"
echo $model
for i in "${arr[@]}"; do
    echo ---------------ontology $i----------------
    echo train.py -ont $i -m $model -dr $data_dir -ep $epoch -d $device
    python train.py -ont $i -m $model -dr $data_dir -ep $epoch -d $device
    echo evaluate.py -m $model -ont $i -dr $data_dir
    python evaluate.py -m $model -ont $i -dr $data_dir
done

# CNN
echo "CNN"
for i in "${arr[@]}"; do
    echo ---------------ontology $i----------------
    echo train_cnn.py -ont $i -dr $data_dir -ep $epoch -d $device
    python train_cnn.py -ont $i -dr $data_dir -ep $epoch -d $device
    echo evaluate.py -m deepgocnn -ont $i -dr $data_dir
    python evaluate.py -m deepgocnn -ont $i  -dr $data_dir
done

# DGG
echo "DGG"
for i in "${arr[@]}"; do
    echo ---------------ontology $i----------------
    echo train_dgg.py -ont $i -dr $data_dir -ep $epoch -d $device
    python train_dgg.py -ont $i -dr $data_dir -ep $epoch -d $device
    echo evaluate.py -m dgg -ont $i -dr $data_dir
    python evaluate.py -m dgg -ont $i  -dr $data_dir
done

model="deepgogat"
echo $model
for i in "${arr[@]}"; do
    echo ---------------ontology $i----------------
    echo train_gat.py -m $model -ont $i -dr $data_dir -ep $epoch -d $device
    python train_gat.py -m $model -ont $i -dr $data_dir -ep $epoch -d $device
    echo evaluate.py -m $model -ont $i -dr $data_dir
    python evaluate.py -m $model -ont $i -dr $data_dir
done