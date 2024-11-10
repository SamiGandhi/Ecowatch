from brisque import BRISQUE

import cv2
import numpy as np
import os
import re
import util
import quality_metrics as metrics


def extract_blocks_from_string(string, roi_blocks):
    # Define a regular expression pattern to match numbers
    pattern = r'(\d+)->(\d+)([A-Za-z])'

    # Search for the pattern in the input string
    match = re.search(pattern, string)

    if match:
        first_digit = int(match.group(1))  # First set of digits
        second_digit = int(match.group(2)) # Second set of digits
        letter = match.group(3)       # Letter
        
    smaller = min(first_digit, second_digit)
    larger = max(first_digit, second_digit)
  
    # Generate the list of numbers between start and end (inclusive)
    numbers_between = list(range(smaller, larger + 1))
    list_ = []
    if letter == 'N':
        for block in numbers_between:
            list_.append([block,'N'])
        roi_blocks.extend(list_)
    elif letter == 'H':
        for block in numbers_between:
            list_.append([block,'H'])
        roi_blocks.extend(list_)
    elif letter == 'L':
        for block in numbers_between:
            list_.append([block,'L'])
        roi_blocks.extend(list_)
    return roi_blocks


def writeline(trace_file, frameNb, psnr,ssim,brisque_current,brisque_original):
    with open(trace_file, "a") as traceFile:
        traceFile.write(f"{frameNb}\t{psnr}\t{ssim}\t{brisque_current}\t{brisque_original}\n")



def create_trace_file(tracePath):
    frameTraceName = os.path.join(tracePath, "quality_metrics")
    with open(frameTraceName, "w") as traceFile:
        traceFile.write("#frameNb\tPsnr\tSSIM\tBrisque_current\tbrisque_original\n")

# Pattern to match the filenames and extract the sequence number
pattern = r'frame(\d+)\.png'
new_pattern = r'frame_(\d+)\.png'  # New pattern


#All of images here are 288X288

source_directory = 'new_res'

#the five setup names 
setups = ['No_Noise']

setup_dict = {
    #'mjpeg_light':'MPEG\\light\\GOP_12',
    #'mjpeg_medium':'MPEG\\Medium\\GOP_12',
    #'proposed_light':'ROI\\light',
    #'proposed_medium':'ROI\\medium',
    'hevc1':'HEVC1',
    'hevc2':'HEVC2',
    'mrim1':'MRIM1',
    'mrim2':'MRIM2',
    'prog':'PROG_TRANS'
}

noise_dict = {
    'No_Noise': ''
}


videos_dict = {
    1 : 'Black_Duck',
    2 : 'Common_Sandpiper',
    3 : 'Common_Shelduck',
    4 : 'Great_Flamingo',
    5 : 'Plumed_Whistling_Duck',
    6 : 'Wood_Stork'
} 

 

resolution_dict = {
    1 : '72X72',
    2 : '144X144',
    3 : '288X288',
    4 : '432X432',
    5: '576X576'
} 

#Parametrize the vid and the size indexes and vid


for size_index in range(1, 6):
    size_out_base_path = resolution_dict[size_index]

    for key, video_out_base_path in videos_dict.items():

        #we must extract the data of the 4 setups
        for key, noise_directory_name in noise_dict.items():
            directory = os.path.join(source_directory,noise_directory_name)
            directory = os.path.join(directory,video_out_base_path)
            directory = os.path.join(directory,size_out_base_path)

            print(f'dir {directory}')

            #getting the settups directories
            proposed_light_directory = os.path.join(directory,'ROI\\light')
            #create the quality metrics files for all the setteps
            #get the st-packet of roi light as a reference
            st_packet_filepath = os.path.join(proposed_light_directory,'trace/st-packet')
            #captured frame as reference only one time because they are the same
            captured_images = os.path.join(proposed_light_directory,'captured_frames')
            
        

            for key, set_up in setup_dict.items():
                files_with_numbers = []
                print('init the roi and reload it ')
                # Opening and read  trace file
                with open(st_packet_filepath, 'r') as trace_file:
                    trace_content = trace_file.readlines()
                #skip the headers line by setting the index = 1 -> 2nd line
                line_index = 1
                #now skip the lines                           of the first frame
                while line_index < len(trace_content):
                    line = trace_content[line_index]
                    values = line.split('\t')
                    frame_number = values[3]
                    if int(frame_number) != 1:
                        break
                    line_index += 1
                if "mjpeg" in key or "proposed" in key:
                    set_up_directory = os.path.join(directory,set_up)
                    print(f'we are in -> {set_up_directory}')
                    decoded_frames = os.path.join(set_up_directory,'decoded')
                    if os.path.exists(os.path.join(set_up_directory,'quality_metrics')):
                        os.remove(os.path.join(set_up_directory,'quality_metrics'))
                    create_trace_file(set_up_directory)
                    trace_file = os.path.join(set_up_directory,'quality_metrics')
                else:
                    new_dir = os.path.join(source_directory,size_out_base_path)
                    set_up_directory = os.path.join(new_dir,set_up)
                    set_up_directory = os.path.join(set_up_directory,video_out_base_path)
                    print(f'we are in -> {set_up_directory}')

                    decoded_frames = os.path.join(set_up_directory,'compressed_frames')
                    if os.path.exists(os.path.join(set_up_directory,'quality_metrics')):
                        os.remove(os.path.join(set_up_directory,'quality_metrics'))
                    create_trace_file(set_up_directory)
                    trace_file = os.path.join(set_up_directory,'quality_metrics')


                for filename in os.listdir(decoded_frames):
                    if "mjpeg" in key or "proposed" in key:
                        match = re.match(pattern, filename)
                    else:
                        match = re.match(new_pattern, filename)
                    if match:
                        # Extract the sequence number as an integer
                        sequence_number = int(match.group(1))
                        #without the frame number 1
                        if sequence_number == 1:
                            continue
                        # Append the tuple (sequence_number, filename) to the list
                        files_with_numbers.append((sequence_number, filename))
                # Sort the list by sequence_number
                files_with_numbers.sort(key=lambda x: x[0])
                for sequence_number,image_name in files_with_numbers:
                    roi_blocks = []
                    
                    if "mjpeg" in key or "proposed" in key:
                        image_filename = os.path.join(decoded_frames,image_name)
                        captured_image_filename = os.path.join(captured_images,image_name)
                    else:
                        
                        capt_imge = image_name.replace("_", "")

                        # Remove leading zeros from the number part
                        # Assuming the number is always at the end of the string
                        number_part = capt_imge[len("frame"):]  # Extract the number part
                        number_part = number_part.lstrip("0")  # Remove leading zeros

                        # Combine the string back together
                        result = "frame" + number_part
                        image_filename = os.path.join(decoded_frames,image_name)
                        captured_image_filename = os.path.join(captured_images,result)

                    image_ = cv2.imread(image_filename)

                    captured_image_ = cv2.imread(captured_image_filename)
                    image = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
                    captured_image = cv2.cvtColor(captured_image_, cv2.COLOR_BGR2GRAY)
                    while line_index < len(trace_content):
                        line = trace_content[line_index]
                        values = line.split('\t')
                        frame_number = values[3]
                        line_blocks_str = values[6].split(' ')
                        for block_str in line_blocks_str:
                            roi_blocks = extract_blocks_from_string(block_str, roi_blocks)
                        if int(frame_number) != sequence_number:
                            break
                        line_index += 1
                    # Extract the sequence number and convert it to an integer
                    blocks_decoded = []
                    blocks_black = []
                    captured_blocks = []
                    three_channels_blocks = []
                    black_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

                    # Divide the image into 8x8 blocks
                    for row in range(0, image.shape[0], 8):
                        for col in range(0, image.shape[1], 8):
                            block = image[row:row+8, col:col+8]
                            blocks_decoded.append(block)

                    for row in range(0, black_image.shape[0], 8):
                        for col in range(0, black_image.shape[1], 8):
                            block = black_image[row:row+8, col:col+8]
                            blocks_black.append(block)

                    for row in range(0, captured_image.shape[0], 8):
                        for col in range(0, captured_image.shape[1], 8):
                            block = captured_image[row:row+8, col:col+8]
                            captured_blocks.append(block)
                    
                    for row in range(0, image_.shape[0], 8):
                        for col in range(0, image.shape[1], 8):
                            block = image_[row:row+8, col:col+8]
                            three_channels_blocks.append(block)

                    ## now iterate over all the roi blocks and replace them from the image to the black image
                    for roi_block,Level in roi_blocks:
                        try:
                            #extract the roi_block from the image 
                            block = blocks_decoded[roi_block]
                            blocks_black[roi_block] = block
                        except TypeError:
                            if roi_block[1] == 'N' :
                                red_block = np.array([255, 0, 0], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                                blocks_black[roi_block[0]] = red_block
                            elif roi_block[1] == 'H':
                                red_block = np.array([0, 255, 0], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                                blocks_black[roi_block[0]] = red_block
                            elif roi_block[1] == 'L':
                                red_block = np.array([0, 0, 255], dtype=np.uint8) * np.ones((8, 8, 3), dtype=np.uint8)
                                blocks_black[roi_block[0]] = red_block

                    image_height = black_image.shape[0]
                    image_width = black_image.shape[1]

                    for i, block in enumerate(blocks_black):
                        if np.any(block>0):
                            continue
                        else:
                            blocks_black[i] = captured_blocks[i]

                        
                    '''
                    '''
                    #rebuild the black frame and show it to confirm
                    # 
                    # Create an empty image to restore the blocks

                    

                    reconstructed_image  = np.zeros((image_height, image_width), dtype=np.uint8)
                    # Iterate over the blocks and place them in the restored image
                    
                    num_blocks_per_row = image_height // 8  # Divide image width by 8 to get the number of blocks per row

                    for i, block in enumerate(blocks_black):
                        # Calculate the position of the current block
                        row = i // num_blocks_per_row
                        col = i % num_blocks_per_row
                        
                        start_row = row * 8
                        start_col = col * 8
                        block_2d = np.squeeze(block)
                        # Place the block in the corresponding position
                        reconstructed_image[start_row:start_row+8, start_col:start_col+8] = block_2d

                    ## calculate the psrn ssim and brisque

                    ssim = metrics.calculate_ssim(captured_image,reconstructed_image)
                    psnr = metrics.calculate_psnr(captured_image,reconstructed_image)
                    reconstructed_image_3ch = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR)
                    brisque_current = metrics.calculate_brisque(reconstructed_image_3ch)
                    brisque_original = metrics.calculate_brisque(captured_image_)
                    writeline(trace_file,sequence_number,psnr,ssim,brisque_current,brisque_original)
