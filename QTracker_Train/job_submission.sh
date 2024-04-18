#!/bin/bash

# Submit the Event Filter Training Job
#job1_id=$(sbatch jobscript_Event_Filter.sh | awk '{print $4}')

# Submit All-Vertex Finding Job, start once the event filter is done.
#job2_id=$(sbatch --dependency=afterok:$job1_id jobscript_TFA.sh | awk '{print $4}')
#job2_id=$(sbatch jobscript_TFA.sh | awk '{print $4}')

# Submit Z-Vertex Finding Job, start once the event filter is done.
#job3_id=$(sbatch --dependency=afterok:$job1_id jobscript_TFZ.sh | awk '{print $4}')
#job3_id=$(sbatch jobscript_TFZ.sh | awk '{print $4}')

# Submit Target-Vertex Finding Job, start once the event filter is done.
#job4_id=$(sbatch --dependency=afterok:$job1_id jobscript_TFT.sh | awk '{print $4}')
#job4_id=$(sbatch jobscript_TFT.sh | awk '{print $4}')

# Submit the All-Vertex Kinematic and Vertex Reconstruction job once the All-Vertex Finder is done training.
#job5_id=$(sbatch --dependency=afterok:$job2_id jobscript_Kin_All.sh | awk '{print $4}')
job5_id=$(sbatch jobscript_Kin_All.sh | awk '{print $4}')

# Submit the Z-Vertex Kinematic and Vertex Reconstruction job once the Z-Vertex Finder is done training.
#job6_id=$(sbatch --dependency=afterok:$job3_id jobscript_Kin_Z.sh | awk '{print $4}')
job6_id=$(sbatch jobscript_Kin_Z.sh | awk '{print $4}')

# Submit the Target-Vertex Kinematic Reconstruction job once the Target-Vertex Finder is done training.
#job7_id=$(sbatch --dependency=afterok:$job4_id jobscript_Kin_Target.sh | awk '{print $4}')
job7_id=$(sbatch jobscript_Kin_Target.sh | awk '{print $4}')

# Submit the target-dump filter training once everything else is done.
sbatch --dependency=afterok:$job5_id:$job6_id:$job7_id jobscript_Target_Dump_Filter.sh
