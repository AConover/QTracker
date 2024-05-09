#!/bin/bash

# Submit the Event Filter Training Job
job1_id=$(sbatch Training_Jobscripts/jobscript_Event_Filter.sh | awk '{print $4}')

# Submit All-Vertex Finding Job, start once the event filter is done.
job2_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job1_id jobscript_TFA.sh | awk '{print $4}')

# Submit Z-Vertex Finding Job, start once the event filter is done.
job3_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job1_id jobscript_TFZ.sh | awk '{print $4}')

# Submit Target-Vertex Finding Job, start once the event filter is done.
job4_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job1_id jobscript_TFT.sh | awk '{print $4}')

# Submit Dump-Vertex Finding Job, start once the event filter is done.
job5_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job1_id jobscript_TFD.sh | awk '{print $4}')

# Submit the All-Vertex Kinematic and Vertex Reconstruction job once the All-Vertex Finder is done training.
job6_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job2_id jobscript_Kin_All.sh | awk '{print $4}')

# Submit the Z-Vertex Kinematic and Vertex Reconstruction job once the Z-Vertex Finder is done training.
job7_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job3_id jobscript_Kin_Z.sh | awk '{print $4}')

# Submit the Target-Vertex Kinematic Reconstruction job once the Target-Vertex Finder is done training.
job8_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job4_id jobscript_Kin_Target.sh | awk '{print $4}')

# Submit the Target-Vertex Kinematic Reconstruction job once the Target-Vertex Finder is done training.
job9_id=$(sbatch Training_Jobscripts/--dependency=afterok:$job5_id jobscript_Kin_Target.sh | awk '{print $4}')

# Submit the target-dump filter training once everything else is done.
sbatch Training_Jobscripts/--dependency=afterok:$job6_id:$job7_id:$job8_id:$job9_id jobscript_Target_Dump_Filter.sh
