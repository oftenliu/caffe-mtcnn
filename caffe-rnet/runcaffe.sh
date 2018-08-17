#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ulsee/often/caffe-mtcnn/caffe-rnet

set -e
/home/ulsee/often/caffe/build/tools/caffe train  \
	 --solver=/home/ulsee/often/caffe-mtcnn/caffe-rnet/solver.prototxt  