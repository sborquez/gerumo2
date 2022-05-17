#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J baseline
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH --account IRIS-IP007-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=24:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue


#! Do not change:
#SBATCH --partition ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime. 

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load cuda/11.2
#module unload cuda/11.4
module load cudnn/8.1_cuda-11.2

export LD_LIBRARY_PATH=/home/ir-riqu1/TensorRT/TensorRT-6.0.1.5/lib:$LD_LIBRARY_PATH
#! Full path to application executable:
rds_dir="$HOME/rds/rds-iris-ip007"
gerumo_dir="$HOME/gerumo2"
singularity_sif="$HOME/rds/rds-iris-ip007/ir-borq1/gerumo2-fixed2.sif"
script="$gerumo_dir/tools/evaluate_net.py"
application="singularity exec --nv $singularity_sif python $script"

#! Run options for the application:
if [ -z "$experiment" ];
then
    options="--config-file "
else
    options="--config-file $experiment"
fi

if [ -z "$dataset" ];
then
    options="$options"
else
    case $dataset in
        test_gm_full)
            events="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_gd/events"
            telescopes="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_gd/telescopes"
            data_folder="/home/ir-borq1/rds/rds-iris-ip007/ir-niet1/datasets/DL1_Prod5/gamma-diffuse/test"
            ;;

        test_gm_cut1000)
            events="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_gd_cut1000/events"
            telescopes="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_gd_cut1000/telescopes"
            data_folder="/home/ir-borq1/rds/rds-iris-ip007/ir-niet1/datasets/DL1_Prod5/gamma-diffuse/test"
            ;;

        test_g_full)
            events="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_g/events"
            telescopes="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_g/telescopes"
            data_folder="/home/ir-borq1/rds/rds-iris-ip007/ir-niet1/datasets/DL1_Prod5/gamma/test"
            ;;

        test_g_cut1000)
            events="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_g_cut1000/events"
            telescopes="/home/ir-borq1/rds/rds-iris-ip007/ir-riqu1/Prod5-parquets/output_T_g_cut1000/telescopes"
            data_folder="/home/ir-borq1/rds/rds-iris-ip007/ir-niet1/datasets/DL1_Prod5/gamma/test"
            ;;
    esac
    opts="DATASETS.TEST.EVENTS $events DATASETS.TEST.TELESCOPES $telescopes DATASETS.TEST.FOLDER $data_folder"
    options="$options --dataset_name $dataset $opts"
fi

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
