#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/often/often/caffe-mtcnn/caffe-onet

set -e
/home/often/often/caffe/build/tools/caffe train  \
	 --solver=/home/often/often/caffe-mtcnn/caffe-onet/solver.prototxt  