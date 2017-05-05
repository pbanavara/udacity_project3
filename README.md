This is my submission for the Udacity Behaviour cloning project

To train the model:

python3 model.py --epoch 10

To run the simulator:

docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit
python drive.py model.h5

The output images are in the output directory

The video output is in the current/parent directory - video.mp4
