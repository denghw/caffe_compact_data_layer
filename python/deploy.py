import sys
import caffe
import glob
import numpy as np

def parse_argument():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                         help = 'The image file or directory to be processed.')
    parser.add_argument('--model_prototxt',
                         help = 'The deploy prototxt file for the model',
                         default = 'models/deepid2/deploy.prototxt')
    parser.add_argument('--model_file',
                         help = 'The pretrained network model to be used',
                         default = 'models/deepid2/deepid__iter_70000.caffemodel')
    parser.add_argument('--mean_file',
                         help = 'The binary proto type mean file for the image set',
                         default = 'data/casia-webface-all/mean.binaryproto')

    args = parser.parse_args()
    return args


def main():

    args = parse_argument()

    MODEL_FILE = args.model_file
    PROTOTXT_FILE = args.model_prototxt
    MEAN_FILE = args.mean_file

    net = caffe.Net(PROTOTXT_FILE,MODEL_FILE,caffe.TEST)

    caffe.set_mode_gpu()
    if MEAN_FILE.endswith('.binaryproto'):
        blobProto = caffe.io.caffe_pb2.BlobProto()
        blobProto.ParseFromString(open(MEAN_FILE).read())
        mean_data = caffe.io.blobproto_to_array(blobProto)[0]
    elif MEAN_FILE.endswith('.npy'):
        mean_data = np.load(open(MEAN_FILE))
        print mean_data.shape
    else:
        print "Error: bad mean file"
        sys.exit()
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    transformer.set_mean('data',mean_data[:,:227,:227])
    transformer.set_raw_scale('data',255)
#    transformer.set_input_scale('data',255)
    #transformer.set_channel_swap('data',(2,1,0))
    
    num_input = 1
    import os
    if os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + 'jpg')]
        num_input = len(inputs)
    else:
        inputs = [caffe.io.load_image(args.input_file)]
   
    inputs = [caffe.io.resize(input,(227,227)) for input in inputs]
    inputs = np.asarray(inputs)
        
    net.blobs['data'].reshape(num_input,3,227,227)
    inputs = [transformer.preprocess('data',input_i) for input_i in inputs]

    net.blobs['data'].data[...] = np.asarray(inputs)

    out = net.forward()
    for i in range(num_input):
        print 'Predict #' + str(i)
        print out['prob'][i]
#        print np.sort(out['prob'][i])[-10:
#        print np.argsort(out['prob'][i])[-10:]
#        print np.argmax(out['prob'][i])
#        print np.max(out['prob'][i])
if __name__ == '__main__':

    main()
