#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/home/often/often/caffe-mtcnn/caffe-onet
set -e
#GLOG_logtostderr=0 GLOG_log_dir=/home/often/often/caffe-mtcnn/log/onet \
/home/often/often/caffe/build/tools/caffe train  \
	 --solver=/home/often/often/caffe-mtcnn/caffe-onet/solver.prototxt  2>&1  | tee /home/often/often/caffe-mtcnn/log/onet/onet.log  