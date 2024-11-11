import cv2
import numpy as np
import pywt
from __exp_energy_model__ import CAPTURE_E_PER_BLOCK
from __exp_energy_model__ import get_saliency_detection_energy
from __exp_energy_model__ import  get_wavelet_encoding_energy
from __trace__ import write_trace_line as write_trace
from __compress_lenght__ import  compress_image_in_blocks

def load_mobile_net():
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
    return net


# 1. Saliency Detection using MobileNet SSD
def detect_salient_regions_mobilenet(image,net):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    salient_regions = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold for detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            salient_regions.append(box.astype("int"))
    return salient_regions



# 4. Wavelet-Based Compression (Transformation) using PyWavelets
def wavelet_transform(image, wavelet='haar', level=2):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

# 5. Reconstruct Image from Wavelet Coefficients
def wavelet_reconstruct(coeffs, wavelet='haar'):
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)



# 6. Adjust resolution using wavelet transform and calculate energy
def adjust_resolution_with_energy(subregions, salient_regions):
    high_res_energy = 0.0
    low_res_energy = 0.0
    total_blocks = 0

    adjusted_subregions = []
    
    for subregion, position in subregions:
        height, width = subregion.shape[:2]
        
        block_count = (height // 8) * (width // 8)  # Assuming 8x8 blocks for capture energy calculation
        total_blocks += block_count
        
        if is_salient(position, salient_regions):
            coeffs = wavelet_transform(subregion)
            high_res_energy += get_wavelet_encoding_energy(subregion.shape)
            adjusted_subregions.append(wavelet_reconstruct(coeffs))
        else:
            low_res_subregion = cv2.resize(subregion, (width // 2, height // 2))
            coeffs = wavelet_transform(low_res_subregion)
            low_res_subregion = wavelet_reconstruct(coeffs)
            low_res_subregion = cv2.resize(low_res_subregion, (width, height))  # Resize back to original size
            low_res_energy += get_wavelet_encoding_energy(subregion.shape)
            adjusted_subregions.append(low_res_subregion)
    
    capture_energy = total_blocks * CAPTURE_E_PER_BLOCK * 1000  # Capture energy in mJ
    saliency_detection_energy = get_saliency_detection_energy()

    total_energy = high_res_energy + low_res_energy + capture_energy + saliency_detection_energy

    return adjusted_subregions, total_energy

# 8. Helper function: Check if a region is salient
def is_salient(subregion_position, salient_regions):
    start_x, start_y, subregion_width, subregion_height = subregion_position
    end_x = start_x + subregion_width
    end_y = start_y + subregion_height

    for (salient_x1, salient_y1, salient_x2, salient_y2) in salient_regions:
        if (start_x < salient_x2 and end_x > salient_x1 and
            start_y < salient_y2 and end_y > salient_y1):
            return True
    return False

# 9. Split image into subregions
def split_image_into_subregions(image, rows, cols):
    height, width = image.shape[:2]
    subregion_height = height // rows
    subregion_width = width // cols

    subregions = []
    for row in range(rows):
        for col in range(cols):
            start_y = row * subregion_height
            start_x = col * subregion_width
            subregion = image[start_y:start_y + subregion_height, start_x:start_x + subregion_width]
            subregions.append((subregion, (start_x, start_y, subregion_width, subregion_height)))
    return subregions

# 10. Recombine subregions into the final image
# 10. Recombine subregions into the final image
def recombine_subregions(subregions, rows, cols, image_shape):
    height, width = image_shape[:2]
    subregion_height = height // rows
    subregion_width = width // cols

    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    index = 0
    for row in range(rows):
        for col in range(cols):
            start_y = row * subregion_height
            start_x = col * subregion_width
            subregion = subregions[index]

            # Ensure subregion has the same number of channels as the final image
            if subregion.shape[-1] == 4:  # Remove alpha channel if present
                subregion = subregion[:, :, :3]

            # Check if the subregion needs resizing to fit exactly
            subregion = cv2.resize(subregion, (subregion_width, subregion_height))

            final_image[start_y:start_y + subregion_height, start_x:start_x + subregion_width] = subregion
            index += 1
            
    return final_image

# Main function for Wavelet-based method
def wavelet_method(image,compress_file_path,frame_number,trace_file,net):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Step 1: Detect salient regions
    salient_regions = detect_salient_regions_mobilenet(image,net)
    
    # Step 2: Split image into subregions
    rows, cols = 4, 4  # Example grid
    subregions = split_image_into_subregions(image, rows, cols)
    
    # Step 3: Adjust resolution and calculate energy using wavelet transform
    adjusted_subregions, total_energy = adjust_resolution_with_energy(subregions, salient_regions)
    
    # Step 4: Recombine subregions into the final image
    final_image = recombine_subregions(adjusted_subregions, rows, cols, image.shape)
    
    cv2.imwrite(compress_file_path,final_image)
    pixel_values = final_image.flatten()
    # Convert each pixel value to an 8-bit binary string and join them into a single string
    #binary_data = ''.join(format(pixel, '08b') for pixel in pixel_values)
    compressed_data, huffman_codes = compress_image_in_blocks(final_image)
    size_bits = len(compressed_data)
    size_bytes = size_bits // 8
    # Calculate time duration for one frame (in seconds)
    time_duration = 1 / 2
    # Calculate bitrate (bits per second)
    bitrate_bps = size_bits / time_duration
    # Convert bitrate to kbps (1 kbps = 1000 bps)
    bitrate_kbps = bitrate_bps / 1000
    write_trace(trace_file, frame_number, size_bytes, total_energy, total_energy, bitrate_kbps)
# Run the Wavelet-based method
if __name__ == "__main__":
    wavelet_method('detect.jpg')
