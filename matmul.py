#!/usr/bin/env python

import sys
import argparse
import numpy as np
import time
import tensorflow as tf

# https://books.google.fr/books?id=Q0s6Vgb98CQC&lpg=PT212&dq=Python+Cookbook+%22Collecting+a+Bunch+of+Named+Items%22&hl=en&pg=PT213&redir_esc=y#v=onepage&q&f=false
## p183 : printf in python
## p178 : collecting a bunch of name items (C struct like)

def printf(format, *args):
    sys.stdout.write(format % args)

class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    
class Matmul:
    def __init__(self, gpu, shape, steps):
        sMatmul       = Struct(gpu="", shape=(0,0))
        sMatmul.gpu   = gpu
        sMatmul.shape = (shape, shape)    
        self.sMatmul  = sMatmul
        self.steps    = steps
        self.time     = 0

    def compute(self):
        # computation
        start = time.time()

        for i in range(self.steps):
            printf("[%d]\r", i)
            random_matrix = tf.random.uniform(shape=self.sMatmul.shape, minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        # https://www.tensorflow.org/guide/eager
        ## tf.matmul can return before completing the matrix multiplication
        ## (e.g., can return after enqueing the operation on a CUDA stream).
        ## The x.numpy() call below will ensure that all enqueued operations
        ## have completed (and will also copy the result to host memory,
        ## so we're including a little more than just the matmul operation
        ## time).
        ## Converts a tensor object into an numpy.ndarray object.
        ## This implicitly means that the converted tensor will be now processed on the CPU.

        _ = sum_operation.numpy()
        
        end = time.time();
        return end - start;
        
    def process(self):
        # run on GPU, if available (tF v2 default option)
        if self.sMatmul.gpu:
            if tf.config.list_physical_devices("GPU"):
                with tf.device("/gpu:0"):
                    self.time = self.compute()
            else:
                print("GPU not found")
        else:
            with tf.device("/cpu:0"):
                self.time = self.compute()
        
    def stat(self):
        # log duration
        if self.sMatmul.gpu:
            printf("Computation duration on GPU: %f\n", self.time)
        else:
            printf("Computation duration on CPU: %f\n", self.time)

def main(args):
    matmul = Matmul(args.gpu, args.shape, args.steps)
    matmul.process()
    matmul.stat()

if __name__ =="__main__":
    # arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", dest="gpu", help="indicate computation device", action="store_true")
    parser.add_argument("-s", "--shape"          , help="squared matrix shape"       , required=True, type=int)
    parser.add_argument("-i", "--steps"          , help="number of steps"            , required=True, type=int)

    args = parser.parse_args()   
    main(args);
