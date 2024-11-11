from ultralytics import YOLO
import re
import os
import cv2

EXIT_FAILURE = -1
def makeDir(dirName):
    try:
        os.makedirs(dirName, exist_ok=True)
        print(f"Directory {dirName} created")
    except OSError as e:
      print(f"Error while creating directory {dirName}")
      exit(EXIT_FAILURE)

def deleteFolderContent(dirName):
    try:
        for filename in os.listdir(dirName):
            filepath = os.path.join(dirName, filename)
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
    except Exception as e:
        print(f"Error while deleting folder content in {dirName}: {e}")


def createResultFile(output_directory):
    yolo_repport = os.path.join(output_directory, "yolo_repport")
    with open(yolo_repport, "w") as y_r:
        y_r.write("frameNb\tresolution\tobject_classNb\tconfidence\tfalse_positive\n")

def create_boxes_file(output_directory,frame_number):
    yolo_repport = os.path.join(output_directory, f'boxes_{frame_number}')
    with open(yolo_repport, "w") as y_r:
        y_r.write("")

def create_boxes_number_file(output_directory):
    yolo_repport = os.path.join(output_directory, 'boxes_number')
    with open(yolo_repport, "w") as y_r:
        y_r.write("frame_nb\tresolution\tbox_number\n")


def write_in_result_file(output_directory,frame_nb,resolution,object_class,confidence,false_positive):

    with open(output_directory + "/yolo_repport", "a") as result_file:
        if not result_file:
            print(f"Problem opening file ... {result_file}/yolo_repport")
            return
        result_file.write(f"{frame_nb}\t{resolution}\t{object_class}\t{confidence}\t{false_positive}\n")

def write_in_boxes_number_file(output_directory,frame_nb,resolution,boxes_number):

    with open(output_directory + "/boxes_number", "a") as result_file:
        if not result_file:
            print(f"Problem opening file ... {result_file}/boxes_number")
            return
        result_file.write(f"{frame_nb}\t{resolution}\t{boxes_number}\n")

def write_in_boxes_file(output_directory,frame_number,x1,y1,x2,y2):

  with open(output_directory + f'/boxes_{frame_number}', "a") as result_file:
      if not result_file:
          print(f"Problem opening file ... {result_file}/boxes_{frame_number}")
          return
      result_file.write(f"{x1} {y1} {x2} {y2}\n")


def ret_frame_number(file_path):
  filename = os.path.basename(file_path)
  # Regular expression to find 'frame' followed by a number
  pattern = r'frame(\d+)'

  # Search for the pattern
  match = re.search(pattern, filename)

  # Extract and print the number if a match is found
  if match:
      number = match.group(1)
      return number
  else:
      return -1


#############################################


def init_yolo_v8_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    return model

#Read the images inside of a directory

source = 'C:\\Users\\Gandhi\\Desktop\\birds'


def analyse_using_yolo(source,model):
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    output_directory = os.path.join(source, 'yolo_results')
    print('--------------------------------------- ')
    print(f'Create results directory : -> {output_directory}')
    deleteFolderContent(output_directory)
    makeDir(output_directory)
    print('--------------------------------------- ')
    print(f'Create output directory for image with boxes : -> {output_directory}')
    images_with_boxes_directory = os.path.join(output_directory,'images_with_boxes')
    deleteFolderContent(images_with_boxes_directory)
    makeDir(images_with_boxes_directory)
    print('--------------------------------------- ')
    print(f'Create trace files in directory directory : -> {output_directory}')
    createResultFile(output_directory)
    create_boxes_number_file(output_directory)
    print('--------------------------------------- \n\n\n')
    correct_classes = [4.0,14.0,33.0]
    # Iterate over the files in the directory
    for filename in os.listdir(source):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Construct the full path to the image
            image_path = os.path.join(source, filename)
            image = cv2.imread(image_path)
            # Run inference on the source
            results = model(image)  # list of Results objects
            frame_number = ret_frame_number(image_path)
            print(f'frame_number -> {frame_number}')
            for r in results:
                boxes = r.boxes
                
                print(f'boxes number -> {len(boxes)} ')
                resolution = r.orig_shape
                resolution_input = f'{resolution[0]}X{resolution[1]}'
                write_in_boxes_number_file(output_directory,frame_number,resolution,len(boxes))
                print(f'resolution -> {resolution[0]}X{resolution[1]}')
                create_boxes_file(output_directory,frame_number)
                for box in boxes:
                    false_positive = True
                    xyxy =  box.xyxy.tolist()
                    x1 = xyxy[0][0]
                    y1 = xyxy[0][1]
                    x2 = xyxy[0][2]
                    y2 = xyxy[0][3]
                    write_in_boxes_file(output_directory,frame_number,int(x1),int(y1),int(x2),int(y2))
                    print('x,y -> ')
                    print(x1)
                    print(y1)
                    print(x2)
                    print(y2)
                    # Draw the rectangle
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    class_ = box.cls.item()
                    for correct in correct_classes:
                        if(class_ == correct):
                            false_positive = False
                            break
                    print(f'class -> {class_} and its {not false_positive}')
                    conf = box.conf.item()
                    print('confidence -> ')
                    print(conf)
                    write_in_result_file(output_directory,frame_number,resolution,class_,conf,false_positive)

            # Display the image
            cv2.imwrite(os.path.join(images_with_boxes_directory,f'frame{frame_number}.jpg'),image)