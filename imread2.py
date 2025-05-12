import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV for bilateral filter

# --- Your original functions (get_upper_threshold, get_lower_threshold, process_image, enhance_image) ---
# These are kept here for completeness if you want to compare, but the new method will be used.
def get_upper_threshold(hist):
    non_zero_indices = hist > 0
    N = hist[non_zero_indices]
    L = len(N) 
    n_window = 5 
    half_n = n_window // 2
    local_maxima = []
    if L > 0: 
        for s in range(L):
            win_start = max(0, s - half_n)
            win_end = min(L, s + half_n + 1) 
            window = N[win_start:win_end]
            center_val = N[s]
            if center_val == np.max(window) and np.sum(window == center_val) == 1:
                local_maxima.append(center_val)
        if not local_maxima: 
            Threshold_upper = np.mean(N) 
        else:
            Threshold_upper = np.mean(local_maxima) 
    else: 
        Threshold_upper = 0 
    # print(f"Calculated Threshold_upper (T_UP): {Threshold_upper}")
    return Threshold_upper

def get_lower_threshold(hist,upper_threshold):
    non_zero_indices = hist > 0
    N = hist[non_zero_indices]
    L = len(N) 
    total_pixels = np.sum(hist)
    if L > 0:
        sta = min(total_pixels, upper_threshold * L)
        threshold_lower = sta / 65536 # This division might make it very small
                                      # Original DPHE often keeps thresholds in terms of pixel counts
                                      # Let's assume it's intended to be a relative threshold or a typo
                                      # For the purpose of this exercise, we'll use the new method primarily.
                                      # If it's a count, it should be `sta` directly.
    else: 
        threshold_lower = 0 
    # print(f"Calculated Threshold_lower (T_LOW): {threshold_lower}") 
    return threshold_lower
    
def process_image(image,hist,upper_threshold,lower_threshold):
    modified_hist = np.copy(hist).astype(np.float64) 
    modified_hist[hist >= upper_threshold] = upper_threshold
    # In original DPHE, lower_threshold is a count. If your lower_threshold is small float due to /65536:
    # The condition `hist <= lower_threshold` might not work as expected if lower_threshold is e.g. < 1
    # Assuming lower_threshold is a count for this logic:
    # For this example, let's assume lower_threshold is a direct count from `sta`.
    # If `threshold_lower` from `get_lower_threshold` is already scaled, this part needs adjustment.
    # For now, assuming it's a count (remove /65536 in get_lower_threshold if so for DPHE)
    effective_lower_threshold_count = lower_threshold # If it was scaled, this would be e.g. lower_threshold * 65536
    modified_hist[(hist > 0) & (hist <= effective_lower_threshold_count)] = effective_lower_threshold_count

    cdf = np.cumsum(modified_hist)
    cdf_max = cdf[-1] 
    if cdf_max == 0: # Avoid division by zero if modified_hist is all zeros
        lut = np.zeros_like(cdf, dtype=np.uint8)
    else:
        lut = np.floor((255 * cdf) / cdf_max) 
    lut = np.clip(lut, 0, 255)            
    lut = lut.astype(np.uint8)          
    enhanced_image_8bit = lut[image]
    return enhanced_image_8bit

def enhance_image(image):
    hist, bins = np.histogram(image.flatten(), bins=65536, range=[0, 65536])
    # plt.figure(figsize=(10, 6))
    # plt.plot(bins[:-1], hist, lw=0.5)
    # plt.title('Original 16-bit Image Histogram (Old Method)')
    # plt.xlabel('Gray Level (0-65535)')
    # plt.ylabel('Pixel Count')
    # plt.grid(True)
    # plt.show()
    upper_threshold = get_upper_threshold(hist)
    # Assuming get_lower_threshold returns a count, not scaled value.
    # If it was scaled by /65536, then process_image needs to know that.
    # For simplicity, let's assume it's a count.
    lower_threshold_count = get_lower_threshold(hist, upper_threshold) 
    # If get_lower_threshold divides by 65536, then:
    # lower_threshold_val = get_lower_threshold(hist,upper_threshold)
    # lower_threshold_count = lower_threshold_val # this would be wrong if it's meant to be a count
    
    enhanced_image = process_image(image, hist, upper_threshold, lower_threshold_count)

    # hist_enhanced, bins_enhanced = np.histogram(enhanced_image.flatten(), bins=256, range=[0, 256])
    # plt.figure(figsize=(10, 6))
    # plt.plot(bins_enhanced[:-1], hist_enhanced, lw=0.5, color='r')
    # plt.title('Enhanced 8-bit Image Histogram (Old Method)')
    # plt.xlabel('Gray Level (0-256)')
    # plt.ylabel('Pixel Count')
    # plt.grid(True)
    # plt.show()
    return enhanced_image
# --- End of original functions ---


def durand_tone_mapping(image_16bit, 
                        sigma_spatial_base=30, 
                        sigma_range_base=0.6, 
                        base_compression_s=0.5,
                        detail_boost=1.0):
    """
    Tone maps a 16-bit image to 8-bit using a method inspired by Durand and Dorsey (2002).

    Args:
        image_16bit (np.ndarray): Input 16-bit image (e.g., uint16).
        sigma_spatial_base (float): Spatial standard deviation for the bilateral filter
                                    applied to the log-luminance image to get the base layer.
                                    Controls the spatial extent of smoothing (in pixels).
                                    Larger values mean more smoothing over larger areas.
        sigma_range_base (float): Range/intensity standard deviation for the bilateral filter.
                                  Controls how much tonal variations are smoothed out (in log units).
                                  Larger values mean pixels with greater intensity differences are still smoothed.
        base_compression_s (float): Compression factor 's' for the base layer (typically s < 1 to reduce contrast).
                                    From B_compressed = s * B (+ offset). The 'offset' part is
                                    handled implicitly by the final normalization in this implementation.
        detail_boost (float): Factor to multiply the detail layer by. Default 1.0 (no change).
                              Values > 1 enhance details, < 1 suppress them.

    Returns:
        np.ndarray: Tone-mapped 8-bit image (uint8).
    """
    if image_16bit.dtype != np.uint16:
        print(f"Warning: Input image dtype is {image_16bit.dtype}, not uint16. Attempting conversion.")
        if np.issubdtype(image_16bit.dtype, np.floating):
            if image_16bit.max() <= 1.0 and image_16bit.min() >= 0.0: # Assume normalized float [0,1]
                 image_16bit = (image_16bit * 65535).astype(np.uint16)
            else: # Assume float with larger range or unknown range
                 image_16bit = np.clip(image_16bit, 0, 65535).astype(np.uint16)
        else: # Other integer types
            image_16bit = image_16bit.astype(np.uint16) # Direct cast, might lose precision or wrap around

    # 0. Ensure input is float32 for calculations
    L_input_float = image_16bit.astype(np.float32)

    # 1. Logarithmic transformation: L_log = log(L_input)
    # Using log1p(x) = log(1+x) for better precision with small L_input_float values
    # This means our "epsilon" is effectively 1.
    L_log = np.log1p(L_input_float) 

    # 2. Image decomposition: Base layer B = BilateralFilter(L_log)
    # cv2.bilateralFilter input needs to be float32.
    # d: Diameter of each pixel neighborhood. If non-positive (e.g., -1), it's computed from sigmaSpace.
    # sigmaColor is our sigma_range_base, sigmaSpace is our sigma_spatial_base.
    L_log_contiguous = np.ascontiguousarray(L_log, dtype=np.float32) # OpenCV might need contiguous array
    B = cv2.bilateralFilter(L_log_contiguous, d=-1, 
                            sigmaColor=sigma_range_base, 
                            sigmaSpace=sigma_spatial_base)

    # 3. Detail layer: D = L_log - B
    D = L_log - B
    
    # Optional: Detail Boost
    if detail_boost != 1.0:
        D = D * detail_boost

    # 4. Base layer compression: B_compressed = s * B + offset
    # As per the prompt, a simple way is B_compressed = s * B.
    # The 'offset' is implicitly handled by the final normalization of L_output_linear.
    # A more complex 'offset' could be calculated to target specific mean brightness,
    # but for this implementation, we use the simpler approach.
    B_compressed = B * base_compression_s
    
    # 5. Image reconstruction (in log domain): L_output_log = B_compressed + D
    L_output_log = B_compressed + D
    # This can be rewritten as: L_output_log = (base_compression_s * B) + (L_log - B)
    #                                      = L_log + (base_compression_s - 1) * B

    # 6. Exponential transformation: L_output = exp(L_output_log)
    # Using expm1(x) = exp(x) - 1, which is the inverse of log1p(x)
    L_output_linear = np.expm1(L_output_log) 
    
    # Ensure no negative values (can happen if L_output_log is very small, making expm1 result near -1)
    L_output_linear[L_output_linear < 0] = 0

    # 7. Normalize to 8-bit range [0, 255] and convert to uint8
    # Option 1: Simple min-max normalization
    min_val = np.min(L_output_linear)
    max_val = np.max(L_output_linear)

    if max_val > min_val:
        # Normalize to [0, 1] then scale to [0, 255]
        image_8bit_float = ((L_output_linear - min_val) / (max_val - min_val)) * 255.0
    else: # Image is flat (all pixels same value)
        # Set to a mid-gray or black depending on the value
        image_8bit_float = np.full_like(L_output_linear, 128 if min_val > 0 else 0, dtype=np.float32)
    
    # Option 2: Percentile clipping (can be more robust to extreme outliers)
    # low_perc, high_perc = np.percentile(L_output_linear, [0.5, 99.5]) # e.g., clip 0.5% from each end
    # L_output_clipped = np.clip(L_output_linear, low_perc, high_perc)
    # min_val_c = np.min(L_output_clipped)
    # max_val_c = np.max(L_output_clipped)
    # if max_val_c > min_val_c:
    #     image_8bit_float = ((L_output_clipped - min_val_c) / (max_val_c - min_val_c)) * 255.0
    # else:
    #     image_8bit_float = np.full_like(L_output_linear, 128 if min_val_c > 0 else 0, dtype=np.float32)
        
    return image_8bit_float.astype(np.uint8)


if __name__ == "__main__":
    # --- Please replace with actual parameters ---
    file_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    # file_path = r'\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw' # Another example
    width = 1280
    height = 1024
    pixel_dtype = np.uint16

    bytes_per_pixel = np.dtype(pixel_dtype).itemsize
    frame_size_bytes = width * height * bytes_per_pixel

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(frame_size_bytes)
            if len(raw_data) < frame_size_bytes:
                print(f"Error: Not enough data in file. Expected {frame_size_bytes}, got {len(raw_data)}")
                exit()
            image_16bit = np.frombuffer(raw_data, dtype=pixel_dtype).reshape((height, width))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit()

    # --- Parameters for Durand Tone Mapping (TUNE THESE FOR YOUR IMAGES) ---
    # sigma_spatial_base: Controls spatial extent of base layer smoothing. 
    #                     Larger values (e.g., 20-100) smooth larger features.
    #                     Try values like 2% to 5% of image smaller dimension.
    # sigma_range_base: Controls intensity smoothing in base layer (log units).
    #                   Larger values (e.g., 0.4-1.0) smooth more tones.
    #                   Range of L_log is approx 0 to log(65536)~11.
    # base_compression_s: Compresses contrast of base layer (0.2-0.8 typically).
    #                     Smaller values = more compression.
    # detail_boost: Enhances or suppresses details (e.g., 0.8-1.5).

    # Example parameter set (start with these and adjust):
    param_set_name = "Default"
    s_spatial = 30     # e.g., for 1024 height, 3% is ~30
    s_range = 0.5      # Mid-range for log values
    s_compress = 0.6   # Moderate compression
    d_boost = 1.0      # No detail boost initially

    # # Example parameter set 2 (stronger base smoothing, more compression for very high dynamic range)
    # param_set_name = "Stronger Smoothing"
    # s_spatial = 50
    # s_range = 0.8
    # s_compress = 0.4
    # d_boost = 1.1 # Slight detail boost

    print(f"\nProcessing with Durand Tone Mapping (Parameters: {param_set_name})...")
    print(f"  sigma_spatial_base = {s_spatial}")
    print(f"  sigma_range_base = {s_range}")
    print(f"  base_compression_s = {s_compress}")
    print(f"  detail_boost = {d_boost}")

    enhanced_image_durand = durand_tone_mapping(image_16bit.copy(), # Use a copy
                                                sigma_spatial_base=s_spatial,
                                                sigma_range_base=s_range,
                                                base_compression_s=s_compress,
                                                detail_boost=d_boost)
    
    # --- Displaying Results ---
    plt.figure(figsize=(18, 6)) # Wider figure for 3 plots
    
    # Original 16-bit (windowed for better visibility)
    plt.subplot(1, 3, 1)
    # Calculate robust min/max for display, e.g., 1st and 99th percentile
    vmin_disp = np.percentile(image_16bit, 1)
    vmax_disp = np.percentile(image_16bit, 99)
    plt.imshow(image_16bit, cmap='gray', vmin=vmin_disp, vmax=vmax_disp)
    plt.title('Original 16-bit (Windowed Display)')
    plt.axis('off')

    # Result from Durand Tone Mapping
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_image_durand, cmap='gray')
    plt.title(f'Durand Tone Mapped (8-bit)\nsp={s_spatial}, sr={s_range}, sc={s_compress}, db={d_boost}')
    plt.axis('off')
    
    # Result from original DPHE method (optional, for comparison)
    # print("\nProcessing with original DPHE method for comparison...")
    # enhanced_image_original_dphe = enhance_image(image_16bit.copy())
    # plt.subplot(1, 3, 3)
    # plt.imshow(enhanced_image_original_dphe, cmap='gray')
    # plt.title('Original DPHE Method (8-bit)')
    # plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Display histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # For 16-bit, a full 65536 bin histogram is too sparse to plot well.
    # Plotting with fewer bins or log scale for y-axis can be better.
    hist_orig, bins_orig = np.histogram(image_16bit.flatten(), bins=256, range=[np.min(image_16bit), np.max(image_16bit)+1])
    plt.plot(bins_orig[:-1], hist_orig, lw=1)
    plt.title('Original 16-bit Image Histogram (256 bins)')
    plt.xlabel(f'Gray Level (Range: {np.min(image_16bit)}-{np.max(image_16bit)})')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.yscale('log') # Log scale often better for HDR image histograms

    plt.subplot(1, 2, 2)
    hist_durand, bins_durand = np.histogram(enhanced_image_durand.flatten(), bins=256, range=[0, 256])
    plt.plot(bins_durand[:-1], hist_durand, lw=1, color='r')
    plt.title('Durand Tone Mapped 8-bit Histogram')
    plt.xlabel('Gray Level (0-255)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
