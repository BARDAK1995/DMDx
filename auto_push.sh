#!/bin/bash

# Set the loop to run for 2 minutes (4 iterations of 30 seconds)
for i in {1..4}
do
    # Add all changes to git
    git add .

    # Commit the changes with a message
    git commit -m "Auto-commit simulation results for debug iteration $i"

    # Push the changes to the remote repository
    git push origin main  # Change 'main' to the correct branch name if needed

    # Wait for 30 seconds before the next commit and push 1800 for 6 hours
    sleep 30
done
