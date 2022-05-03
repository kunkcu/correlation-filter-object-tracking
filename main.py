import cv2 as cv
import os
import time
import numpy as np
from tracker import *


class FrameReader:
    def __init__(self, sequence_path):
        self.sequence_path = sequence_path
        self.index = 0

        if os.path.isdir(self.sequence_path):
            # Input mode: image sequence
            _, _, self.image_files = next(os.walk(self.sequence_path))
            self.size = len(self.image_files)

            self.read_function = self.read_image_sequence_frame
        else:
            # Input mode: video
            self.video_capture = cv.VideoCapture(self.sequence_path)
            self.size = self.video_capture.get(cv.CAP_PROP_FRAME_COUNT)

            self.read_function = self.read_video_frame

    def read_image_sequence_frame(self):
        curr_frame = None

        if self.index < self.size:
            # Read current frame
            curr_frame = cv.imread(os.path.join(self.sequence_path, self.image_files[self.index]))

            self.index = self.index + 1

        return curr_frame

    def read_video_frame(self):
        curr_frame = None

        if self.index < self.size and self.video_capture.isOpened():
            # Read current frame
            ret, curr_frame = self.video_capture.read()

            self.index = self.index + 1
        else:
            self.video_capture.release()

        return curr_frame

    def read_frame(self):
        return self.read_function()

    def good(self):
        return self.index < self.size


def reinitalise_tracker(curr_frame, curr_frame_gray, params, ground_truth_path):
    if ground_truth_path is None:
        # Prompt for ROI selection
        bb = np.array(cv.selectROI('Tracker', curr_frame, False, False), dtype=int)
        target_bounding_box = BoundingBox(bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3])
    else:
        true_target_bounding_box = []

        with open(ground_truth_path, 'r') as f:
            for line in f:
                numbers = line.split()
                numbers = numbers[1:5]

                for number in numbers:
                    true_target_bounding_box.append(int(number))

                break

        target_bounding_box = BoundingBox((true_target_bounding_box[0] + true_target_bounding_box[2]) / 2, (true_target_bounding_box[1] + true_target_bounding_box[3]) / 2,
                                          true_target_bounding_box[2] - true_target_bounding_box[0], true_target_bounding_box[3] - true_target_bounding_box[1])

    # Get image shape
    height, width, _ = curr_frame.shape

    # Initialise tracker
    tracker = Tracker(params)
    tracker.initialise(curr_frame_gray, target_bounding_box)

    return params, tracker, target_bounding_box


def test_tracker(sequence_path, params, save, ground_truth_path):
    # Initialise test sequence reader
    frame_reader = FrameReader(sequence_path)
    frame_counter = 0

    tracker = None
    target_bounding_box = None
    video_writer = None
    target_bounding_boxes = []
    track_total_time = 0.0

    while(frame_reader.good()):
        # Read current frame
        curr_frame = frame_reader.read_frame()

        # Convert to grayscale
        curr_frame_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY).astype(np.float32)

        # If first frame
        if frame_counter == 0:
            # Initialise tracker for the first time
            params, tracker, target_bounding_box = reinitalise_tracker(curr_frame, curr_frame_gray, params, ground_truth_path)

            # Initialise video writer (if enabled)
            if save:
                video_writer = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (curr_frame.shape[1], curr_frame.shape[0]))

            fps = 0.0
        else:
            track_start_time = time.time()

            # Get current target localisation
            target_bounding_box = tracker.track(curr_frame_gray)

            track_end_time = time.time()

            # Calculate FPS
            track_total_time = track_total_time + (track_end_time - track_start_time)
            fps = 1 / (track_end_time - track_start_time)

        # Increment frame counter
        frame_counter = frame_counter + 1

        # Visualise current frame and target localization
        curr_frame_disp = np.copy(curr_frame)
        xtl = int(target_bounding_box.xc - target_bounding_box.w / 2)
        ytl = int(target_bounding_box.yc - target_bounding_box.h / 2)
        xbr = int(xtl + target_bounding_box.w)
        ybr = int(ytl + target_bounding_box.h)

        cv.rectangle(curr_frame_disp, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

        # Display FPS
        cv.putText(curr_frame_disp, 'FPS={:0.2f}'.format(fps), org=(15, 15), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=(0, 0, 255), thickness=1)

        cv.imshow('Tracker', curr_frame_disp)
        key = cv.waitKey(33)

        # Save current frame (if enabled)
        if save:
            video_writer.write(curr_frame_disp)

        # Key functionalities
        if key == ord('q'):
            # Exit
            break
        elif key == ord('r'):
            # Reinitialise tracker
            params, tracker, target_bounding_box = reinitalise_tracker(curr_frame, curr_frame_gray, params, ground_truth_path)

        xtl = int(target_bounding_box.xc - target_bounding_box.w / 2)
        ytl = int(target_bounding_box.yc - target_bounding_box.h / 2)
        target_bounding_boxes.append([xtl, ytl, target_bounding_box.w, target_bounding_box.h])

    # Close video writer
    if video_writer is not None:
        video_writer.release()

    # Close windows
    cv.destroyAllWindows()

    print('Mean FPS = ' + '{:0.2f}'.format(frame_counter / track_total_time))

    return target_bounding_boxes


def measure_tracker_performance(predTargetBoundingBoxes, groundTruthPath):
    trueTargetBoundingBoxes = []

    with open(groundTruthPath, 'r') as f:
        for line in f:
            trueTargetBoundingBox = []

            numbers = line.split()
            numbers = numbers[1:5]

            for number in numbers:
                trueTargetBoundingBox.append(int(number))

            temp = trueTargetBoundingBox
            trueTargetBoundingBox = [(temp[0] + temp[2])/2, (temp[1] + temp[3])/2,
                                     temp[2] - temp[0], temp[3] - temp[1]]

            trueTargetBoundingBoxes.append(trueTargetBoundingBox)

    avgAcc_iou = 0  # Metric 1: Average accuracy found using intersection-over-union
    eao = 0  # Metric 2: Expected Average Overlap
    number_of_eao_curves = 10
    eao_vector = np.zeros(number_of_eao_curves)

    noOfFrames = len(predTargetBoundingBoxes)

    for iFrame in range(noOfFrames):
        predTargetBoundingBox = predTargetBoundingBoxes[iFrame]
        trueTargetBoundingBox = trueTargetBoundingBoxes[iFrame]

        dx = min(predTargetBoundingBox[0] + predTargetBoundingBox[2], trueTargetBoundingBox[0] + trueTargetBoundingBox[2]) - \
            max(predTargetBoundingBox[0], trueTargetBoundingBox[0])

        dy = min(predTargetBoundingBox[1] + predTargetBoundingBox[3], trueTargetBoundingBox[1] + trueTargetBoundingBox[3]) - \
            max(predTargetBoundingBox[1], trueTargetBoundingBox[1])

        if dx > 0 and dy > 0:
            intersection = dx * dy
        else:
            intersection = 0

        union = predTargetBoundingBox[2] * predTargetBoundingBox[3] + trueTargetBoundingBox[2] * trueTargetBoundingBox[3] - intersection
        acc_forIthFrame_iou = intersection / union
        avgAcc_iou += acc_forIthFrame_iou / noOfFrames

        if iFrame >= (noOfFrames - number_of_eao_curves):
            eao_vector[iFrame - (noOfFrames - number_of_eao_curves)] = avgAcc_iou

    eao = np.sum(eao_vector) / number_of_eao_curves

    print('Accuracy w.r.t intersection-over-union = ' + '{:0.3f}'.format(avgAcc_iou))
    print('EAO = ' + '{:0.3f}'.format(eao))


def main():
    sequence_path = 'sample/helicopter'
    ground_truth_path = None

    # Get tracker parameters
    params = TrackerParameters(resized_size=(224,224), feature_type='intensity', sigma=3.0, search_area_scale_factor=2.0, scale_factors=[0.98, 1.00, 1.02], lr=0.15)

    # Test tracker
    pred_target_bounding_boxes = test_tracker(sequence_path=sequence_path, params=params, save=True, ground_truth_path=ground_truth_path)

    # Measure tracker performance
    if ground_truth_path is not None:
        measure_tracker_performance(pred_target_bounding_boxes, ground_truth_path)

main()