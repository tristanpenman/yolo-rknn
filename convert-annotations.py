# pylint: disable=invalid-name

import os
import sys

import cv2
import numpy as np

from tqdm import tqdm

ROOT_DIR = os.getcwd()


# function that turns XMin, YMin, XMax, YMax coordinates to normalized yolo format
def convert(path, coords):
    os.chdir("..")
    image = cv2.imread(path + ".jpg")
    coords[2] -= coords[0]
    coords[3] -= coords[1]
    x_diff = int(coords[2]/2)
    y_diff = int(coords[3]/2)
    coords[0] = coords[0]+x_diff
    coords[1] = coords[1]+y_diff
    coords[0] /= int(image.shape[1])
    coords[1] /= int(image.shape[0])
    coords[2] /= int(image.shape[1])
    coords[3] /= int(image.shape[0])
    os.chdir("Label")
    return coords


def process_dir(current_dir, classes):
    os.chdir(current_dir)
    print("Currently in subdirectory: ", current_dir)
    class_dirs = os.listdir(os.getcwd())

    # for all class folders step into directory to change annotations
    for class_dir in class_dirs:
        if os.path.isdir(class_dir):
            os.chdir(class_dir)
            print("Converting annotations for class: ", class_dir)

            # Step into Label folder where annotations are generated
            os.chdir("Label")

            for filename in tqdm(os.listdir(os.getcwd())):
                filename_str = str.split(filename, ".")[0]
                if filename.endswith(".txt"):
                    annotations = []
                    with open(filename, encoding="utf-8") as f:
                        for line in f:
                            for class_type in classes:
                                line = line.replace(class_type, str(classes.get(class_type)))
                            labels = line.split()
                            coords = np.asarray([float(labels[1]),
                                                float(labels[2]),
                                                float(labels[3]),
                                                float(labels[4])])
                            coords = convert(filename_str, coords)

                            labels[1:5] = coords[0:4]

                            newline = str(labels[0]) + " " \
                                    + str(labels[1]) + " " \
                                    + str(labels[2]) + " " \
                                    + str(labels[3]) + " " \
                                    + str(labels[4])

                            line = line.replace(line, newline)
                            annotations.append(line)
                        f.close()
                    os.chdir("..")
                    with open(filename, "w", encoding="utf-8") as outfile:
                        for line in annotations:
                            outfile.write(line)
                            outfile.write("\n")
                        outfile.close()
                    os.chdir("Label")
            os.chdir("..")
            os.chdir("..")

    os.chdir("..")


def main():
    args = sys.argv[1:]

    # Map each string to its index
    classes = {arg: idx for idx, arg in enumerate(args)}

    if len(classes) == 0:
        # create dict to map class names to numbers for yolo
        classes = {}
        with open("classes.txt", "r", encoding="utf-8") as myFile:
            for num, line in enumerate(myFile, 0):
                line = line.rstrip("\n")
                classes[line] = num
            myFile.close()

    # step into dataset directory
    os.chdir(os.path.join("OID", "Dataset"))
    dirs = os.listdir(os.getcwd())

    # for all train, validation and test folders
    for current_dir in dirs:
        if os.path.isdir(current_dir):
            process_dir(current_dir, classes)


if __name__ == "__main__":
    main()
