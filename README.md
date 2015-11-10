# caffe_compact_data_layer
It add a CompactDataLayer to implement more data transformation operations

This is another linux version of real-time augmentation based on @ChenlongChen's [caffe-windows](https://github.com/ChenglongChen/caffe-windows)

@senecaur's [caffe-rta](https://github.com/senecaur/caffe-rta) is another importance reference. 

The motivation of this new version is the modifications in the original caffe framwork's APIs which makes
it hard to merge into our current work.

The definition of the layer is almost identical to that of caffe-windows in spite of some minimal changes, including

- eliminate the multiscale parameter, and take it as default
- eliminate the jpeg-compression
- debug-display will save a jpg image instead of showing it because our server runs in command-line mode.

The setting of other parameters can be referred from [caffe-windows](https://github.com/ChenglongChen/caffe-windows). Also,
a template is located in the directory of models/test/train_val.prototxt.  
