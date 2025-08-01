# StarDist Nucleus Segmentation Script
# This script processes individual H&E images to detect and segment nuclei using StarDist

import os
import sys
from imageio.v3 import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import device
import tensorflow as tf

def calculate_optimal_parameters(min_dimension):
    """
    Calculate optimal StarDist parameters based on image dimensions.
    
    This function implements adaptive parameter selection to avoid AssertionError
    while maximizing nucleus detection quality. Different strategies are used
    based on image size to ensure optimal processing.
    
    Args:
        min_dimension (int): The smaller dimension of the image (width or height)
    
    Returns:
        dict: Dictionary containing optimal parameters for StarDist processing
    """
    
    if min_dimension >= 4096:
        # Large images: use reference parameters from StarDist documentation
        # These parameters work well for high-resolution images
        return {
            'block_size': 4096,
            'min_overlap': 128,
            'context': 128,
            'n_tiles': (4, 4, 1)
        }
    elif min_dimension >= 2048:
        # Medium images: use large blocks with safety margins
        # Calculate block size that fits within image dimensions
        block_size = min(2048, min_dimension - 128)
        min_overlap = max(64, block_size // 16)
        # Use safer context calculation with larger safety margin
        max_safe_context = block_size - min_overlap - 64  # Safety margin to prevent AssertionError
        context = max(64, max_safe_context)
        return {
            'block_size': block_size,
            'min_overlap': min_overlap,
            'context': context,
            'n_tiles': (2, 2, 1)
        }
    else:
        # Small images: use conservative parameters to avoid crashes
        # This handles images smaller than 2048px safely
        block_size = min(1024, min_dimension - 64)
        min_overlap = max(32, block_size // 8)
        
        # Use safer context calculation for small images
        max_safe_context = block_size - min_overlap - 32  # Safety margin
        context = max(32, max_safe_context)  # Lower minimum for small images
        
        # Determine optimal tile configuration based on image size
        if min_dimension >= 1024:
            n_tiles = (2, 2, 1)  # 2x2 tiles for medium images
        elif min_dimension >= 512:
            n_tiles = (2, 1, 1)  # 2x1 tiles for smaller images
        else:
            n_tiles = (1, 1, 1)  # Single tile for very small images
            
        return {
            'block_size': block_size,
            'min_overlap': min_overlap,
            'context': context,
            'n_tiles': n_tiles
        }

def run_stardist_on_image(img_path):
    """
    Main function to process a single image with StarDist nucleus segmentation.
    
    This function handles the complete pipeline from image loading to output generation,
    including error handling and parameter optimization.
    
    Args:
        img_path (str): Path to the input image file
    """
    # Extract base name for output files
    base_name = os.path.basename(img_path).replace(".png", "")
    img = imread(img_path)
    
    # Print image dimensions for debugging and monitoring
    print(f"Image shape: {img.shape}")
    
    # Apply percentile normalization to improve image contrast
    # This helps StarDist detect nuclei more accurately
    min_percentile = 5
    max_percentile = 95
    img_norm = normalize(img, min_percentile, max_percentile)

    # Force CPU-only processing to avoid GPU-related issues on cluster
    # This ensures consistent behavior across different compute nodes
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    
    # Model loading with robust error handling
    # Try local model first, then fallback to pretrained download
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, 'models', '2D_versatile_he')
    
    try:
        if os.path.exists(local_model_path):
            # Use local model copy to avoid network issues across compute nodes
            print(f"Loading local StarDist model from {local_model_path}...")
            model = StarDist2D(None, name='2D_versatile_he', basedir=os.path.join(script_dir, 'models'))
            print("Successfully loaded local '2D_versatile_he' model")
        else:
            # Fallback to downloading from StarDist if local copy unavailable
            print("Loading StarDist2D '2D_versatile_he' model from pretrained...")
            model = StarDist2D.from_pretrained('2D_versatile_he')
            print("Successfully loaded pretrained '2D_versatile_he' model")
    except Exception as e:
        # If model loading fails, skip this image rather than crashing the job
        print(f"ERROR: Failed to load '2D_versatile_he' model: {e}")
        print("This compute node has model loading issues. Skipping this image.")
        return

    # Calculate optimal parameters based on image dimensions
    # This prevents AssertionError while maximizing detection quality
    height, width = img.shape[:2]
    min_dimension = min(height, width)
    
    print(f"Image dimensions: {height}x{width}, min dimension: {min_dimension}")
    
    # Get optimal parameters for this image size
    params = calculate_optimal_parameters(min_dimension)
    print(f"Optimal parameters: block_size={params['block_size']}, min_overlap={params['min_overlap']}, context={params['context']}, n_tiles={params['n_tiles']}")
    
    # Process image with StarDist using calculated parameters
    with device('/CPU:0'):
        try:
            # Run nucleus segmentation with optimal parameters
            labels, _ = model.predict_instances_big(
                img_norm,
                axes='YXC',  # Y=height, X=width, C=channels
                block_size=(params['block_size'], params['block_size'], 3),
                prob_thresh=0.01,  # Probability threshold for nucleus detection
                nms_thresh=0.001,  # Non-maximum suppression threshold
                min_overlap=(params['min_overlap'], params['min_overlap'], 0),
                context=(params['context'], params['context'], 0),
                normalizer=None,  # Already normalized above
                n_tiles=params['n_tiles']
            )
            print("Successfully processed with optimal parameters")
        except AssertionError as e:
            # If optimal parameters fail, use minimal safe parameters
            print(f"AssertionError with optimal parameters: {e}")
            print("Trying with minimal parameters...")
            
            # Fallback to very conservative parameters
            block_size = min(256, min_dimension - 32)
            min_overlap = max(16, block_size // 16)
            context = max(16, block_size // 16)
            
            print(f"Minimal parameters: block_size={block_size}, min_overlap={min_overlap}, context={context}, n_tiles=(1,1,1)")
            
            # Try again with minimal parameters
            labels, _ = model.predict_instances_big(
                img_norm,
                axes='YXC',
                block_size=(block_size, block_size, 3),
                prob_thresh=0.01,
                nms_thresh=0.001,
                min_overlap=(min_overlap, min_overlap, 0),
                context=(context, context, 0),
                normalizer=None,
                n_tiles=(1, 1, 1)
            )

    # Save nucleus segmentation mask
    # Each nucleus has a unique integer ID (0 = background, 1,2,3... = nuclei)
    np.save(f"{base_name}_labels.npy", labels)

    # Create visualization overlay showing nuclei on original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')  # Original image in grayscale
    plt.imshow(labels, cmap='jet', alpha=0.4)  # Nuclei overlay in color
    plt.title(f"Segmentation Overlay: {base_name}")
    plt.axis('off')
    plt.savefig(f"{base_name}_overlay.png", bbox_inches='tight')
    plt.close()

    # Print completion message with nucleus count
    print(f"Finished processing: {base_name}")
    print(f"Detected {len(np.unique(labels))-1} nuclei")  # -1 because 0 is background

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) < 2:
        print("Usage: python run_stardist.py <image_path>")
        sys.exit(1)
    run_stardist_on_image(sys.argv[1])
