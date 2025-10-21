#!/bin/bash
cd /path/to/netflix-show-clustering

while true; do
    git add .
    git commit -m "Auto-sync: $(date)"
    git push origin main
    sleep 300  
done
