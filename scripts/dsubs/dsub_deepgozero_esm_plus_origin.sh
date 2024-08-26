#!/bin/bash
#DSUB -n DeepGOZero_ESMPLUS_ORIGIN
#DSUB -A root.project.P24Z10200N0983
#DSUB -R 'cpu=64;gpu=1;mem=10000'
#DSUB -N 1
#DSUB -eo /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/deepgo2/outputs/logs/DeepGOZero_ESMPLUS_ORIGIN.%J.%I.err
#DSUB -oo /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/deepgo2/outputs/logs/DeepGOZero_ESMPLUS_ORIGIN.%J.%I.out

## must Edit DSUB -oo & -eo file path

## Set scripts
RANK_SCRIPT="./scripts/tasks/deepgozero_esm_plus_origin.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/deepgo2"

## Set NNODES
NNODES=5
GPUS=2

## Create nodefile

JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/outputs/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/outputs/tmp/${JOB_ID}.nodefile
cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
#cat ${CCS_ALLOC_FILE} > :q!${JOB_PATH}/outputs/tmp/CCS_ALLOC_FILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${GPUS} ${NODEFILE}
