import numpy as np
import cv2 as cv
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model


def get_cosine_window(height, width):
    window_row = np.expand_dims(np.hanning(height), axis=1)
    window_col = np.expand_dims(np.hanning(width), axis=0)

    window = window_row @ window_col

    return window


def get_gaussian_response_function(height, width, sigma):
    y = np.expand_dims(np.arange(height), axis=1)
    x = np.expand_dims(np.arange(width), axis=0)

    gaussian_resp_row = np.exp(-np.square(y - height / 2) / (2 * np.square(sigma)))
    gaussian_resp_col = np.exp(-np.square(x - width / 2) / (2 * np.square(sigma)))

    gaussian_resp = gaussian_resp_row @ gaussian_resp_col

    return gaussian_resp


def get_preprocessed_sample(sample, window):
    height, width = sample.shape
    sample = np.log(sample + 1)

    # Normalise image
    sample = (sample - np.mean(sample)) / (np.std(sample) + 1e-5)

    # Apply windowing on image
    sample = sample * window

    return sample


def get_cropped_sample(frame, bounding_box, curr_scale, params):
    w = bounding_box.w * curr_scale * params.search_area_scale_factor
    h = bounding_box.h * curr_scale * params.search_area_scale_factor

    # Calculate sample boundaries
    x_tl = int(bounding_box.xc - w / 2)
    x_br = int(x_tl + w)
    y_tl = int(bounding_box.yc - h / 2)
    y_br = int(y_tl + h)

    # Calculate required padding
    if x_tl < 0:
        x_l_pad = -x_tl
    else:
        x_l_pad = 0

    if x_br > frame.shape[1] - 1:
        x_r_pad = x_br - frame.shape[1]
    else:
        x_r_pad = 0

    if y_tl < 0:
        y_t_pad = - y_tl
    else:
        y_t_pad = 0

    if y_br > frame.shape[0] - 1:
        y_b_pad = y_br - frame.shape[0]
    else:
        y_b_pad = 0

    # Add required padding to the frame
    frame_padded = np.pad(frame, [(y_t_pad, y_b_pad), (x_l_pad, x_r_pad)], 'edge')

    # Crop sample
    sample = frame_padded[y_tl+y_t_pad:y_br+y_t_pad,x_tl+x_l_pad:x_br+x_l_pad]

    # Resize sample to the desired size
    sample_resized = cv.resize(sample, params.resized_size, interpolation=cv.INTER_AREA)

    return sample_resized


def get_extracted_features(frame, bounding_box, window, curr_scale, params):
    features = None

    # Get cropped sample
    sample = get_cropped_sample(frame, bounding_box, curr_scale, params)

    if params.feature_type == 'intensity':
        # Preprocess sample
        features = get_preprocessed_sample(sample, window)
    elif params.feature_type == 'gradient':
        # Preprocess sample
        sample = get_preprocessed_sample(sample, window)

        features = np.gradient(np.gradient(sample, axis=0), axis=1)
    elif params.feature_type == 'hog':
        # Preprocess sample
        sample = get_preprocessed_sample(sample, window)

        _, features = hog(sample, orientations=9, pixels_per_cell=(8,8),
                                cells_per_block=(1,1), visualize=True, multichannel=False, block_norm='L2')
    elif params.feature_type == 'deep':
        # Repeat intensity channel to RGB
        sample = np.expand_dims(sample, axis=(0,3))
        sample = np.repeat(sample, 3, axis=3)

        # Preprocess sample
        sample = preprocess_input(sample)

        # Extract features
        features = np.squeeze(params.model.predict(sample))

        # Resize features to the desired size
        features = cv.resize(features, params.resized_size, interpolation=cv.INTER_AREA)

        # Normalise features
        features = (features - np.mean(features)) / (np.std(features) + 1e-5)

        # Apply windowing on features
        features = features * np.repeat(np.expand_dims(window, axis=2), features.shape[2], axis=2)

    # Calculate feature Fourier
    features_fourier = np.fft.fft2(features,axes=(0,1))
    features_fourier_conjugate = np.conjugate(features_fourier)

    return features, features_fourier, features_fourier_conjugate


# x_top_left, y_top_left, width, height
class BoundingBox:
    def __init__(self, xc, yc, w, h):
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h


class TrackerParameters:
    def __init__(self, resized_size, feature_type, sigma, search_area_scale_factor, scale_factors, lr):
        self.resized_size = resized_size
        self.feature_type = feature_type
        self.sigma = sigma
        self.search_area_scale_factor = search_area_scale_factor
        self.scale_factors = scale_factors
        self.lr = lr

        if self.feature_type == 'deep':
            # Create model to extract deep features
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 4))])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)

            # InceptionV3
            self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.resized_size[0], self.resized_size[0], 3))
            self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[5].output)

            # VGG16
            # self.model = VGG16(weights='imagenet', include_top=False, input_shape=(self.params.resized_size[0], self.params.resized_size[0], 3))
            # self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-5].output)
        else:
            self.model = None


class Tracker:
    def __init__(self, params):
        self.params = params

        # Get cosine window (used in normalisation)
        self.window = get_cosine_window(self.params.resized_size[0], self.params.resized_size[1])

        # Generate desired gaussian response function and its Fourier
        self.gi = get_gaussian_response_function(self.params.resized_size[0], self.params.resized_size[1], self.params.sigma)
        self.Gi = np.fft.fft2(self.gi)

    def initialise(self, frame, bounding_box):
        self.bounding_box = bounding_box
            
        # Get extracted features
        _, Fi, Fic =  get_extracted_features(frame, self.bounding_box, self.window, 1.0, self.params)

        # Reshape gaussian response function fourier (if required)
        if Fi.ndim > 2:
            self.Gi = np.repeat(np.expand_dims(self.Gi, axis=2), Fi.shape[2], axis=2)

        # Initialise filter parts
        self.Ai = self.Gi * Fic
        self.Bi = Fi * Fic

    def track(self, frame):
        best_peak_score = 0
        best_scale = -1
        y_disp = 0
        x_disp = 0
        tracking_failure = False

        # PSR (Peak-to-Sidelobe Ratio) threshold for tracking failure detection
        psr_threshold = 2

        for iScale, scale in enumerate(self.params.scale_factors):
            # Get extracted features
            _, Fi, Fic =  get_extracted_features(frame, self.bounding_box, self.window, scale, self.params)

            # Get response map and its space form
            Gr = (self.Ai / self.Bi) * Fi

            if Fi.ndim > 2:
                Gr = np.sum(Gr, axis=2)

            gr = np.real(np.fft.ifft2(Gr))

            # Find peak position
            peak_score = np.max(gr)
            peak_position = np.where(gr == peak_score)

            if peak_score > best_peak_score:
                best_peak_score = peak_score
                best_scale = scale

                y_disp = int(np.mean(peak_position[0]) - gr.shape[0] / 2)
                x_disp = int(np.mean(peak_position[1]) - gr.shape[1] / 2)

                # Find PSR for the given scale (with 21x21 sidelobe, 5x5 central region)
                gr_padded = np.pad(gr, [(10, 10), (10, 10)], 'edge')
                peak_position_padded = (int(peak_position[0] + 10), int(peak_position[1] + 10))
                sidelobe_vector = np.concatenate((gr_padded[peak_position_padded[0]-10:peak_position_padded[0]-5, peak_position_padded[1]-10:peak_position_padded[1]+10].flatten(),
                                                gr_padded[peak_position_padded[0]+5:peak_position_padded[0]+10, peak_position_padded[1]-10:peak_position_padded[1]+10].flatten(),
                                                gr_padded[peak_position_padded[0]-10:peak_position_padded[0]+10, peak_position_padded[1]-10:peak_position_padded[1]-5].flatten(),
                                                gr_padded[peak_position_padded[0]-10:peak_position_padded[0]+10, peak_position_padded[1]+5:peak_position_padded[1]+10].flatten()))
                sidelobe_mean = np.mean(sidelobe_vector)
                sidelobe_std = np.std(sidelobe_vector)

                if (peak_score - sidelobe_mean) / sidelobe_std < psr_threshold:
                    tracking_failure = True
                else:
                    tracking_failure = False

        # Calculate resize multipliers
        y_scale_mul = self.bounding_box.h * best_scale * self.params.search_area_scale_factor / self.params.resized_size[0]
        x_scale_mul = self.bounding_box.w * best_scale * self.params.search_area_scale_factor / self.params.resized_size[1]

        # Update target position
        self.bounding_box.yc = self.bounding_box.yc + y_disp * y_scale_mul
        self.bounding_box.xc = self.bounding_box.xc + x_disp * x_scale_mul

        # Update target scale
        self.bounding_box.h = best_scale * self.bounding_box.h
        self.bounding_box.w = best_scale * self.bounding_box.w

        if not tracking_failure:
            # Get extracted features (again with found localization)
            _, Fi, Fic =  get_extracted_features(frame, self.bounding_box, self.window, 1.0, self.params)

            # Update filter
            self.Ai = self.params.lr * (self.Gi * Fic) + (1 - self.params.lr) * self.Ai
            self.Bi = self.params.lr * (Fi * Fic) + (1 - self.params.lr) * self.Bi
        
        return self.bounding_box