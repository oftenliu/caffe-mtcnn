#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ulsee/often/caffe-mtcnn/caffe-pnet

set -e
/home/often/often/caffe/build/tools/caffe train  \
	 --solver=/home/often/often/caffe-mtcnn/caffe-pnet/solver.prototxt  