#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <starting_number> <highest_number>"
    exit 1
fi

starting_number=$1
highest_number=$2

for ((i=$starting_number; i<=$highest_number; i++)); do
    # Create a new directory with the name of the number
    dir_name="$i"
    mkdir "$dir_name"
    
    # Copy files to the new directory
    cp jobscript.sh QTracker_Rivanna.py list_spill_good.txt "$dir_name"
    
    # Change the first line of the copied QTracker_Rivanna.py
    sed -i "s|root_directory = '/project/ptgroup/seaquest/data/digit/02/'|root_directory = '/project/ptgroup/seaquest/data/digit/02/$i/'|" "$dir_name/QTracker_Rivanna.py"
    
    # Grant execute permissions to jobscript.sh
    chmod +x "$dir_name/jobscript.sh"
    
    # Run jobscript.sh in the new directory
    (cd "$dir_name" && sbatch ./jobscript.sh) &
done

# Wait for all background jobs to finish
wait

