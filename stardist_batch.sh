#!/bin/bash

# StarDist Batch Processing Script
# This script submits SLURM jobs to process multiple H&E images in parallel
# It includes smart job management to avoid reprocessing completed images

# Define directory paths for the pipeline
IMAGEDIR="/gpfs/commons/groups/innovation/jjoseph/004/highres"  # Input images directory
SCRIPT="/gpfs/commons/groups/innovation/jjoseph/004/run_stardist.py"  # Main processing script
LOGDIR="/gpfs/commons/groups/innovation/jjoseph/004/logs"  # SLURM job logs directory

# Create logs directory if it doesn't exist
mkdir -p "$LOGDIR"

# Function to check if an image was successfully processed this prevents redundant processing and saves computational resources


is_processed() {
    local base_name="$1"
    
    # Check if both labels and overlay files exist
    if [[ -f "${base_name}_labels.npy" && -f "${base_name}_overlay.png" ]]; then
        return 0  # Successfully processed
    else
        return 1  # Not processed or failed
    fi
}

# Main processing loop iterate through all PNG images in the input directory
for img in "$IMAGEDIR"/*.png; do

    # Extract base name from image filename (remove .png extension)
    base=$(basename "$img" .png)
    
    # Skip if already successfully processed
    # This prevents wasting computational resources on completed images
    
    if is_processed "$base"; then
        echo "Skipping $base - already processed successfully"
        continue
    fi
    
    echo "Submitting job for $base"

    # Submit SLURM job for this image
    # Each job runs independently on a compute node
    
    sbatch <<EOF
    
#!/bin/bash
#SBATCH -J smurf_${base}                    # Job name (unique for each image)
#SBATCH --partition=cpu                     # Use CPU partition (no GPU needed)
#SBATCH --mem=64G                          # Request 64GB memory per job
#SBATCH --cpus-per-task=8                  # Request 8 CPU cores per job
#SBATCH --time=2-00:00:00                  # Time limit: 2 days
#SBATCH --output=$LOGDIR/${base}_%j.out    # Standard output log file
#SBATCH --error=$LOGDIR/${base}_%j.err     # Error output log file
#SBATCH --mail-type=END,FAIL               # Email notifications for job completion/failure
#SBATCH --mail-user=jjoseph@nygenome.org   # Email address for notifications

# Load required modules for the environment
module purge                               # Clear any loaded modules
module load miniconda3                    

# Activate the conda environment with all required packages
eval "\$(conda shell.bash hook)"          # Initialize conda for this shell


conda activate /gpfs/commons/groups/innovation/jjoseph/enact_results/intexgrate_env

# Set up environment variables for the processing
export PYTHONPATH="/gpfs/commons/groups/innovation/jjoseph/smurf_results/SMURF:\${PYTHONPATH:-}"  # Add SMURF to Python path
export MPLBACKEND=Agg                    
export CUDA_VISIBLE_DEVICES=-1            # Disable GPU usage (force CPU processing)

# Run the StarDist processing script on this image
python "$SCRIPT" "$img"
EOF

done
