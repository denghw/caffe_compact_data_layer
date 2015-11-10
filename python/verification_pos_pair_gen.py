import lmdb
import numpy as np
import caffe    
import caffe.io as io
import caffe.proto.caffe_pb2 as pb

def lmdb_read_ini(db_name):
    db = lmdb.open(db_name)
    txn = db.begin()
    cur = txn.cursor()
    cur.iternext()  #After initialization, the curosr is at an unpositional state
    return txn,cur

def lmdb_write_ini(db_name):
    env = lmdb.Environment(db_name,map_size=1099511627776)
    txn = env.begin(write=True)
    return txn

def read_data(cur):
    if cur.key == '':
        cur.first()
        print 'Key is Null'
    datum = pb.Datum()
    datum.ParseFromString(cur.value())
    return datum

def pair_gen(datum_1,datum_2):
    mat_1 = io.datum_to_array(datum_1)
    mat_2 = io.datum_to_array(datum_2)
    mat_3 = np.concatenate([mat_1,mat_2],axis=0)
    datum_3 = io.array_to_datum(mat_3)
    if datum_1.label == datum_2.label:
        datum_3.label = 1
    else:
        datum_3.label = 0
    return datum_3


def iter_one_step(cur):
    if not cur.next():
        cur.first()
    return cur

def verification_pair_gen(input_db_name,output_db,pair_num,mode=0):
    
    txn_r,cur_r = lmdb_read_ini(input_db_name)
    txn_w = lmdb_write_ini(output_db)
    change_flag = 1
    for pair_id in range(pair_num):
        rnd_1 = int(np.floor(np.random.randn()*20))
        rnd_2 = int(np.floor(np.random.randn()*20))
        if change_flag == 1:
            for i in range(rnd_1):
                if not cur_r.next(): cur_r.first()
            datum_1 = read_data(cur_r)
            change_flag = 0
        if not cur_r.next(): cur_r.first()
        datum_2 = read_data(cur_r)
        if mode == 1:
            while datum_2.label != datum_1.label:
                if not cur_r.next(): 
                    cur_r.first()
                    change_flag = 1
                datum_2 = read_data(cur_r)
        else:
            while datum_2.label == datum_1.label:
                if not cur_r.next(): 
                    cur_r.first()
                    change_flag = 1
                datum_2 = read_data(cur_r)
        datum_3 = pair_gen(datum_1,datum_2)
        value = datum_3.SerializeToString()
        txn_w.put('%08d'%pair_id,value)
        if pair_id % 1000 == 0 and pair_id != 0:
            print '%d pairs have been generated'%pair_id
            txn_w.commit()
            txn_w = lmdb_write_ini(output_db)
        else:
            print '%d pairs have been generated'%pair_id
    if pair_num % 1000 != 0:
        print '%d pairs have been generated'%pair_num
        txn_w.commit()
    
if __name__ == '__main__':
   verification_pair_gen('data/casia-webface-all/casia-all-train-lmdb/','data/casia-webface-all/casia-ver-pos-train-lmdb',300000,1) 
