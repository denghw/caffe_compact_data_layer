import lmdb
import numpy as np
import theano
import theano.tensor as T
def lmdb_initialize(db_name):
    db = lmdb.open(db_name)
    txn = db.begin()
    cur = txn.cursor()
    cur.iternext()  #After initialization, the curosr is at an unpositional state
    return (txn,cur)

def load_data_lmdb(cur,batch_size):
    data = []
    label = []
    if cur.key == '':
        cur.first()
    import caffe.io as io
    import caffe.proto.caffe_pb2 as pb
    datum = pb.Datum()
    for i in range(batch_size):
        value = cur.value()
        datum.ParseFromString(value)
        img_mat = io.datum_to_array(datum)
        shape = (datum.channels,datum.height,datum.width)
        data.append(img_mat.reshape(np.prod(shape),))
        label.append(datum.label)
        if not cur.next():
            cur.first()
    data  = np.asarray(data,dtype = theano.config.floatX)
    label = np.asarray(label,dtype= 'int32')
    return (data,label)

if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        print ""
    db_name = sys.argv[1]
    txn,cur = lmdb_initialize(db_name)
    for i in range(n_batch):
        data,label = load_data_lmdb(cur,sys.argv[2])

