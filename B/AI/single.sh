#!/bin/bash
dnn="${dnn:-resnet20}"
bs="${bs:-32}"
epoch="${epoch:-1}"
iter="${iter:-800}"
secs="${secs:--1}"
cuda="${cuda:-1}"
source exp_configs/$dnn.conf
nstepsupdate=1
python -W ignore  \
    dl_trainer.py \
    --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --num-iter $iter --secs $secs --cuda $cuda --power-profile 
# dnn=lstman4
# source exp_configs/$dnn.conf
# nworkers=4
# nwpernode=1
# nstepsupdate=1
# MPIPATH=/home/comp/zhtang/openmpi3.1.1
# $MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#     -mca pml ob1 -mca btl ^openib \
#     -mca btl_tcp_if_include bond0 \
#     -x NCCL_P2P_DISABLE=1 \
#     python horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode 



