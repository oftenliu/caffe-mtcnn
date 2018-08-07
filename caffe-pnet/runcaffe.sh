#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ulsee/often/caffe-mtcnn/caffe-pnet

set -e
/home/ulsee/often/caffe/build/tools/caffe train  \
	 --solver=/home/ulsee/often/caffe-mtcnn/caffe-pnet/solver.prototxt  