#!/bin/bash

# Set the loop to run for 6 hours (12 iterations of 30 minutes)
for i in {1..12}
do
    # Navigate to the directory where your folder is located
    cd /path/to/your/folder

    # Add all changes to git
    git add .

    # Commit the changes with a message
    git commit -m "Auto-commit simulation results at iteration $i"

    # Push the changes to the remote repository
    git push origin main  # Change 'main' to the correct branch name if needed

    # Wait for 30 minutes (1800 seconds) before the next commit and push
    sleep 600
done
