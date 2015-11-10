import numpy as np
import caffe
import sys
import os

def cPickle_output(vars, file_name):
    print '\twriting data to %s' % (file_name)
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Usage: feature_extract.py num_files device result_dir"
        sys.exit()
    num_files = int(sys.argv[1])
    print type(num_files)
    print num_files
    mini_batch = 1000
    n_batches = num_files / mini_batch
    left_batch = num_files - n_batches * mini_batch 
    caffe.set_mode_gpu()
    caffe.set_device(int(sys.argv[2]))
    net = caffe.Net('models/casia_mnist/casia_train_val.prototxt','models/casia_mnist/_iter_400000.caffemodel',caffe.TEST)
    result_dir = sys.argv[3]
    if not result_dir.endswith('/'):
        result_dir += '/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for i in range(n_batches):
        net.forward()
        x = net.blobs['pool5'].data
        y = net.blobs['label'].data
        x.shape = (mini_batch,-1)
        result_file = result_dir + str(i)
        cPickle_output((x,y),result_file)
    if left_batch > 0 :
        net.forward()
        x = net.blobs['pool5'].data[:left_batch]
        y = net.blobs['label'].data[:left_batch]
        result_file = result_dir + str(n_batches)
        cPickle_output((x,y),result_file)
