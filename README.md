# Correlation Filter Object Tracking

TODO: Add more information about the project

## Authors:
* Burak Künkçü ([@burakkunkcu](https://github.com/burakkunkcu/))
* Ceyhun Emre Öztürk ([@ceyhunemreozturk](https://github.com/ceyhunemreozturk))

## Requirements:
* Python environment (virtualenv, conda etc.)
* Python packages:
    * numpy (>= 1.18)
    * scikit-image (>= 0.18.1)
    * tensorflow (>= 2.5.0)
    * opencv-python (>= 4.5.2.52)

## Usage
* Activate python environment (also ensure that required packages are installed)
* Download sample data: `bash download_sample_data.sh`
* Tracker parameters are defined in main.py
* Run tracker: `python main.py`
* Select bounding box when prompted, press `ENTER`
* When `R` key is pressed, user will be prompted for a new bounding box and program will be reset
* When `Q` key is pressed, program will be terminated

## References
Inspired from https://github.com/TianhongDai/mosse-object-tracking
