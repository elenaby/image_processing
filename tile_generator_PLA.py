
import tifffile
import numpy as np
import os
from skimage.filters import threshold_otsu
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

# Configuration
# input_path = r"Z:\Leica Cell Dive\Antibody optimization\CK7 [EPR1619Y]\Titration\Cntrl Kidney 22 CK7 EPR1619Y 1-300 Titration Texas Red\Scan1\Cntrl Kidney 22 CK7 EPR1619Y 1-300 Titration Texas Red_Scan1.qptiff"
# output_dir = r"Z:\Computational Team\Safrygina\CellDive\selected_tiles"

input_path = r"Z:\TRIALS IHC-mIF images\PLA\PLA TEST 5 IF 15-08-2025\PLA T5 IF 15-08-2025\Cntrl Lymph Node 3 PD1-PDL1 PLA T5 opal 690\Scan1.qptiff"
output_dir = r"Z:\Computational Team\Safrygina\PLA\Cntrl_Lymph_Node_3_PD1_PDL1_PLA_T5_opal_690_Scan_1"

tile_size = 256
min_foreground_pixels = 5000  # Reduced threshold since tissue is sparse
max_tiles_to_keep = 100

os.makedirs(output_dir, exist_ok=True)

print("Reading QPTIFF file...")
print("=" * 60)

def extract_texture_features(image):
    """Extract texture features from an image tile"""
    if image.max() == image.min():
        return np.zeros(6)
    
    # Convert to uint8 for texture analysis
    if image.dtype != np.uint8:
        img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        img_norm = image.copy()
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(img_norm, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract texture properties
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop)[0, 0])
    
    return np.array(features)

def calculate_similarity(features1, features2):
    """Calculate similarity between two feature vectors"""
    # Normalize features
    f1_norm = (features1 - np.mean(features1)) / (np.std(features1) + 1e-8)
    f2_norm = (features2 - np.mean(features2)) / (np.std(features2) + 1e-8)
    
    # Cosine similarity
    similarity = np.dot(f1_norm, f2_norm) / (np.linalg.norm(f1_norm) * np.linalg.norm(f2_norm) + 1e-8)
    return similarity

try:
    with tifffile.TiffFile(input_path) as tif:
        # Read the full resolution 2-channel image
        print("Reading full resolution image...")
        image = tif.asarray(series=0)
        print(f"Full resolution image shape: {image.shape}")
        
        # Extract channels
        nuclei = image[0]  # Channel 0 - Nuclei
        membrane = image[1]  # Channel 1 - Membrane/Cytoplasm
        
        print(f"\nChannel shapes:")
        print(f"Nuclei: {nuclei.shape}, dtype: {nuclei.dtype}")
        print(f"Membrane: {membrane.shape}, dtype: {membrane.dtype}")
        
        # Check pixel statistics
        print(f"\nPixel statistics:")
        print(f"Nuclei - Min: {nuclei.min()}, Max: {nuclei.max()}, Mean: {nuclei.mean():.2f}")
        print(f"Membrane - Min: {membrane.min()}, Max: {membrane.max()}, Mean: {membrane.mean():.2f}")
        
        # STEP 1: First, let's find good tiles using the approach from before
        height, width = nuclei.shape
        print(f"\nImage size: {width} x {height}")
        
        # Enhance contrast for better feature extraction
        print("\nEnhancing contrast for processing...")
        
        def enhance_contrast(channel, low_percentile=2, high_percentile=98):
            """Enhance contrast using percentile stretching"""
            p_low = np.percentile(channel, low_percentile)
            p_high = np.percentile(channel, high_percentile)
            
            if p_high > p_low:
                enhanced = np.clip((channel - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
            else:
                enhanced = channel.astype(np.uint8)
            
            return enhanced
        
        nuclei_enhanced = enhance_contrast(nuclei)
        membrane_enhanced = enhance_contrast(membrane)
        
        print(f"After enhancement:")
        print(f"Nuclei - Min: {nuclei_enhanced.min()}, Max: {nuclei_enhanced.max()}")
        print(f"Membrane - Min: {membrane_enhanced.min()}, Max: {membrane_enhanced.max()}")
        
        # Find initial candidate tiles
        print(f"\nFinding initial candidate tiles...")
        
        candidate_tiles = []
        candidate_positions = []
        candidate_features = []
        
        # Use systematic sampling with some randomness
        np.random.seed(42)
        grid_size = 30  # More granular grid for sparse tissue
        
        for i in range(200):  # Check 200 random positions
            y = np.random.randint(0, height - tile_size)
            x = np.random.randint(0, width - tile_size)
            
            nuclei_tile = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
            
            # Skip if tile is too dark
            if nuclei_tile.mean() < 20:
                continue
            
            try:
                # Use Otsu thresholding
                threshold = threshold_otsu(nuclei_tile)
                binary = nuclei_tile > threshold
                foreground = np.sum(binary)
                
                # Lower threshold for sparse tissue
                if foreground > min_foreground_pixels:
                    # Extract texture features
                    features = extract_texture_features(nuclei_tile)
                    
                    candidate_tiles.append((x, y))
                    candidate_positions.append((x, y))
                    candidate_features.append(features)
                    
            except:
                continue
        
        print(f"Found {len(candidate_tiles)} candidate tiles")
        
        if len(candidate_tiles) == 0:
            print("No candidate tiles found! Trying with even lower threshold...")
            # Try with much lower threshold
            min_foreground_pixels = 1000
            
            for i in range(300):
                y = np.random.randint(0, height - tile_size)
                x = np.random.randint(0, width - tile_size)
                
                nuclei_tile = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
                
                if nuclei_tile.mean() < 10:
                    continue
                
                try:
                    threshold = threshold_otsu(nuclei_tile)
                    binary = nuclei_tile > threshold
                    foreground = np.sum(binary)
                    
                    if foreground > min_foreground_pixels:
                        features = extract_texture_features(nuclei_tile)
                        
                        candidate_tiles.append((x, y))
                        candidate_positions.append((x, y))
                        candidate_features.append(features)
                        
                except:
                    continue
        
        if len(candidate_tiles) == 0:
            print("Still no tiles found. Using top tiles by mean intensity...")
            # Last resort: use top tiles by mean intensity
            intensities = []
            positions = []
            
            for i in range(500):
                y = np.random.randint(0, height - tile_size)
                x = np.random.randint(0, width - tile_size)
                
                nuclei_tile = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
                mean_intensity = nuclei_tile.mean()
                
                intensities.append(mean_intensity)
                positions.append((x, y))
            
            # Get indices of top intensities
            top_indices = np.argsort(intensities)[-50:]  # Top 50
            
            for idx in top_indices:
                x, y = positions[idx]
                nuclei_tile = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
                features = extract_texture_features(nuclei_tile)
                
                candidate_tiles.append((x, y))
                candidate_positions.append((x, y))
                candidate_features.append(features)
        
        # STEP 2: If we have tiles, analyze their similarity
        if len(candidate_tiles) > 1:
            print(f"\nAnalyzing {len(candidate_tiles)} candidate tiles...")
            
            # Calculate all pairwise similarities
            similarity_matrix = np.zeros((len(candidate_tiles), len(candidate_tiles)))
            for i in range(len(candidate_tiles)):
                for j in range(len(candidate_tiles)):
                    if i != j:
                        sim = calculate_similarity(candidate_features[i], candidate_features[j])
                        similarity_matrix[i, j] = sim
            
            # Find most representative tile (highest average similarity to others)
            avg_similarities = np.mean(similarity_matrix, axis=1)
            representative_idx = np.argmax(avg_similarities)
            
            rep_x, rep_y = candidate_positions[representative_idx]
            print(f"Most representative tile at position: ({rep_x}, {rep_y})")
            print(f"Average similarity to other tiles: {avg_similarities[representative_idx]:.3f}")
            
            # Find tiles most similar to the representative
            similarities_to_rep = similarity_matrix[representative_idx, :]
            
            # Sort tiles by similarity to representative
            sorted_indices = np.argsort(similarities_to_rep)[::-1]  # Descending
            
            # Select top tiles
            selected_indices = sorted_indices[:max_tiles_to_keep]
            selected_positions = [candidate_positions[i] for i in selected_indices]
            
            print(f"\nSelected {len(selected_positions)} tiles based on similarity:")
            for i, idx in enumerate(selected_indices):
                x, y = candidate_positions[idx]
                sim_score = similarities_to_rep[idx]
                print(f"  Tile {i}: position=({x},{y}), similarity={sim_score:.3f}")
        
        else:
            # Fallback: use candidate tiles directly
            selected_positions = candidate_positions[:max_tiles_to_keep]
            print(f"\nUsing {len(selected_positions)} candidate tiles")
        
        # STEP 3: Save the selected tiles in multiple formats
        print(f"\n" + "=" * 60)
        print("SAVING TILES")
        print("=" * 60)
        
        for i, (x, y) in enumerate(selected_positions):
            # Extract original tiles (not enhanced)
            nuclei_tile_original = nuclei[y:y + tile_size, x:x + tile_size]
            membrane_tile_original = membrane[y:y + tile_size, x:x + tile_size]
            
            # Create enhanced versions for visualization
            nuclei_tile_enhanced = nuclei_enhanced[y:y + tile_size, x:x + tile_size]
            membrane_tile_enhanced = membrane_enhanced[y:y + tile_size, x:x + tile_size]
            
            # Calculate Otsu threshold for this tile
            try:
                threshold = threshold_otsu(nuclei_tile_enhanced)
                binary = nuclei_tile_enhanced > threshold
                foreground = np.sum(binary)
            except:
                foreground = 0
            
            print(f"\nProcessing tile {i+1} at ({x}, {y}):")
            print(f"  Foreground pixels: {foreground}")
            print(f"  Nuclei range: {nuclei_tile_original.min()}-{nuclei_tile_original.max()}")
            print(f"  Membrane range: {membrane_tile_original.min()}-{membrane_tile_original.max()}")
            
            # Create output filenames
            base_name = f"tile_{i:03d}"
            
            # 1. Save separate nuclei channel (Channel 0)
            nuclei_path = os.path.join(output_dir, f"{base_name}_nuclei.tiff")
            tifffile.imwrite(nuclei_path, nuclei_tile_original)
            print(f"  ✓ Saved nuclei channel: {base_name}_nuclei.tiff")
            
            # 2. Save separate cytoplasm/membrane channel (Channel 1)
            membrane_path = os.path.join(output_dir, f"{base_name}_cytoplasm.tiff")
            tifffile.imwrite(membrane_path, membrane_tile_original)
            print(f"  ✓ Saved cytoplasm channel: {base_name}_cytoplasm.tiff")
            
            # 3. Save 2-channel TIFF (like original)
            two_channel_path = os.path.join(output_dir, f"{base_name}_2channel.tiff")
            two_channel_data = np.stack([nuclei_tile_original, membrane_tile_original], axis=0)
            tifffile.imwrite(
                two_channel_path,
                two_channel_data,
                imagej=True,
                metadata={'axes': 'CYX', 'channels': 2, 'Channel0': 'nuclei', 'Channel1': 'cytoplasm'}
            )
            print(f"  ✓ Saved 2-channel TIFF: {base_name}_2channel.tiff")
            
            # 4. Save RGB preview for visualization
            preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
            preview_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            
            # Normalize each channel for display
            if nuclei_tile_enhanced.max() > 0:
                preview_rgb[..., 0] = (nuclei_tile_enhanced / nuclei_tile_enhanced.max() * 255).astype(np.uint8)
            if membrane_tile_enhanced.max() > 0:
                preview_rgb[..., 1] = (membrane_tile_enhanced / membrane_tile_enhanced.max() * 255).astype(np.uint8)
            
            cv2.imwrite(preview_path, preview_rgb)
            print(f"  ✓ Saved RGB preview: {base_name}_preview.png")
        
        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total tiles processed: {len(selected_positions)}")
        print(f"\nOutput directory: {output_dir}")
        print(f"\nFor each tile, 4 files were created:")
        print(f"  1. [tile]_nuclei.tiff    - Nuclei channel only")
        print(f"  2. [tile]_cytoplasm.tiff - Cytoplasm/Membrane channel only")
        print(f"  3. [tile]_2channel.tiff  - 2-channel composite (like original)")
        print(f"  4. [tile]_preview.png    - RGB preview (red=nuclei, green=cytoplasm)")
        
        # Create a summary file
        summary_path = os.path.join(output_dir, "tile_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Tile Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source: {input_path}\n")
            f.write(f"Tile size: {tile_size}x{tile_size}\n")
            f.write(f"Total tiles: {len(selected_positions)}\n\n")
            
            for i, (x, y) in enumerate(selected_positions):
                nuclei_tile = nuclei[y:y + tile_size, x:x + tile_size]
                membrane_tile = membrane[y:y + tile_size, x:x + tile_size]
                
                f.write(f"Tile {i:03d}:\n")
                f.write(f"  Position: ({x}, {y})\n")
                f.write(f"  Nuclei range: {nuclei_tile.min()}-{nuclei_tile.max()}\n")
                f.write(f"  Cytoplasm range: {membrane_tile.min()}-{membrane_tile.max()}\n")
                f.write(f"  Files:\n")
                f.write(f"    - tile_{i:03d}_nuclei.tiff\n")
                f.write(f"    - tile_{i:03d}_cytoplasm.tiff\n")
                f.write(f"    - tile_{i:03d}_2channel.tiff\n")
                f.write(f"    - tile_{i:03d}_preview.png\n\n")
        
        print(f"\nSummary saved to: {summary_path}")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()