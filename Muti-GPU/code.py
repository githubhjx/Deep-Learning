# Assign somenoe gpu
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# muti-gpu
with tf.decive(('/gpu:0^1^2^3...')
