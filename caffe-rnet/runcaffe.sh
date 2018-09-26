#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/ulsee/often/caffe-mtcnn/caffe-rnet

set -e
#GLOG_logtostderr=0 GLOG_log_dir=/home/often/often/caffe-mtcnn/log/rnet \
/home/often/often/caffe/build/tools/caffe train  \
	 --solver=/home/often/often/caffe-mtcnn/caffe-rnet/solver.prototxt  2>&1  | tee /home/often/often/caffe-mtcnn/log/rnet/rnet.log  