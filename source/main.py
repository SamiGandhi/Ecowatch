import capture as cap
import math
import numpy as np
import util
import main_frame
import second_frame
import trace
import decoder
import roi
import cv2
from parameters import parameters as para
def run():
    seq_nb = [0]
    if para.running == 'decoding':
        util.makeDir(para.decoded_frames_dir)
        frame_nb = cap.capture_frames()
        decoder.build_received_video(frame_nb)
    elif para.running == 'decoding_roi':
        util.makeDir(para.decoded_frames_dir)
        frame_nb = cap.capture_frames() - 1
        decoder.build_received_video_roi(frame_nb)
    elif para.running == 'Roi':
        captured_frames_list = []
        seq_nb = [0]
        util.makeDir(para.output_directory)
        util.makeDir(para.trace_file_path)
        util.makeDir(para.captured_frame_dir)
        util.delete_folder_countent(para.captured_frame_dir)
        util.makeDir(para.reference_frames_dir)
        util.delete_folder_countent(para.reference_frames_dir)
        util.makeDir(para.roi_frame_dir)
        util.createTraceFiles(para.trace_file_path)
        util.delete_folder_countent(para.masks_dir)
        util.makeDir(para.masks_dir)
        frame_nb = cap.capture_frames()
        for i in range(1,frame_nb+1):
            frame_record = trace.frame_record()
            frame_record.frameNb = i
            frame_record.frameType = 'M'
            frame_record.layersSizeVector.clear()
            frame_record.layersSizeVector = [0] * para.level_numbers 
            frame = cap.get_captured_frame(i)
            if i == 1:#first frame is stored as it is
                #
                prev = frame
            else:#retreive the roi
                gray_frame = frame
                sad_map,r_mask, g_mask = roi.get_masks(gray_frame,prev)
                c1, c2, c3 = roi.get_compression_masks(sad_map,r_mask, g_mask)
                cv2.imwrite(f"{para.masks_dir}/sadmap_frame{i}.png", sad_map)
                cv2.imwrite(f"{para.masks_dir}/rmask_frame{i}.png", r_mask)
                cv2.imwrite(f"{para.masks_dir}/gmask_frame{i}.png", g_mask)
                cv2.imwrite(f"{para.masks_dir}/c1_frame{i}.png", c1)
                cv2.imwrite(f"{para.masks_dir}/c2_frame{i}.png", c2)
                cv2.imwrite(f"{para.masks_dir}/c3_frame{i}.png", c3)


                roi_frame = roi.apply_mask(gray_frame,g_mask)
                cv2.imwrite(f"{para.roi_frame_dir}/roi_frame{i}.png",roi_frame)
                main_frame.encode_roi_main_frame(roi_frame,frame_record,i,seq_nb,c1,c2,c3)
                prev = frame
            
            """
            frame_record = trace.frame_record()
            frame_record.frameNb = i
            if i > 1:
                # Calculate the MSE
                if main_frame_.frame.shape != frame.shape:
                    print(f"Error: Frame number: {main_frame_.frame_number} and Frame number: {i} have different dimensions")
                    exit()
                mse = util.getMSE(main_frame_.frame, frame)
                mse = math.sqrt(mse)
                
            if i == 1 or mse > para.gop_coefficient:
                captured_frame = cap.Captured_Frame(frame,i,'M')
                captured_frames_list.append(captured_frame)
                frame_record.frameType = 'M'
                frame_record.layersSizeVector.clear()
                frame_record.layersSizeVector = [0] * para.level_numbers 
                main_frame.encode_main_frame(captured_frame.frame,frame_record,i,seq_nb)
                main_frame_ = captured_frame
                # Encode main frame
            else:
                frame_record.frameType = 'S'
                captured_frame = cap.Captured_Frame(frame,i,'S')
                captured_frame.reference_frame = main_frame_.frame_number
                captured_frames_list.append(captured_frame)
                second_frame.encodeSecondFrame(captured_frame,main_frame_.frame_number,frame_record,seq_nb)
                #Encode S frame
            """
    else : 
        captured_frames_list = []
        util.makeDir(para.output_directory)
        util.makeDir(para.trace_file_path)
        util.makeDir(para.captured_frame_dir)
        util.delete_folder_countent(para.captured_frame_dir)
        util.makeDir(para.reference_frames_dir)
        util.delete_folder_countent(para.reference_frames_dir)
        util.createTraceFiles(para.trace_file_path)
        frame_nb = cap.capture_frames()
        #frame_nb = cap.capture_frames_original_fps()
        for i in range(1,frame_nb+1):
            frame = cap.get_captured_frame(i)
            frame_record = trace.frame_record()
            frame_record.frameNb = i
            if i > 1:
                # Calculate the MSE
                if main_frame_.frame.shape != frame.shape:
                    print(f"Error: Frame number: {main_frame_.frame_number} and Frame number: {i} have different dimensions")
                    exit()
                mse = util.getMSE_2(main_frame_.frame, frame)
                mse = math.sqrt(mse)
            if i == 1 or mse > para.gop_coefficient:
                captured_frame = cap.Captured_Frame(frame,i,'M')
                captured_frames_list.append(captured_frame)
                frame_record.frameType = 'M'
                frame_record.layersSizeVector.clear()
                frame_record.layersSizeVector = [0] * para.level_numbers 
                main_frame.encode_main_frame(captured_frame.frame,frame_record,i,seq_nb)
                main_frame_ = captured_frame
                # Encode main frame
            else:
                frame_record.frameType = 'S'
                captured_frame = cap.Captured_Frame(frame,i,'S')
                captured_frame.reference_frame = main_frame_.frame_number
                captured_frames_list.append(captured_frame)
                second_frame.encodeSecondFrame(captured_frame,main_frame_.frame_number,frame_record,seq_nb)
                #Encode S frame
            

        for captured_frame in captured_frames_list:
            if captured_frame.frame_type == 'M':
                print(f'Frame number: {captured_frame.frame_number} is {captured_frame.frame_type} frame.')
            else:
                print(f'Frame number: {captured_frame.frame_number} is {captured_frame.frame_type} frame, its refrence frame is {captured_frame.reference_frame}.')









