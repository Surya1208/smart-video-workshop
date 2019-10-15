
# Test FP32 /FP16 /TF-TensorRT on Pix2Pix


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

isTRT=True
try:
    import tensorflow.contrib.tensorrt as trt
except ImportError:
    print ('Tensor RT not found. IT will be disable  from the benchamrk')
    isTRT=False

import time
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
from tensorflow.python.tools import freeze_graph

import shutil
import argparse, sys, datetime

tf.logging.set_verbosity(tf.logging.INFO)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

import numpy as np
import tensorflow.contrib.slim as slim


#Balinet Deifivition:
parameters = []

conv_counter = 1
def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType,hole,type,format):
    global conv_counter
    global parameters
    #format =  'NHWC'#'NCHW'
    name = 'conv' + str(conv_counter)
    conv_counter += 1

    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=type,
                                                 stddev=1e-1), name='weights')
        if hole>1:
            dH=1
            dW=1

        if format == 'NCHW':
          strides = [1, 1, dH, dW]
          dilations = [1, 1, hole, hole]
        else:
          strides = [1, dH, dW, 1]
          dilations = [1,hole, hole,1]
        # print(format)
        # print(name)
        # print('dilations:'+str(dilations))

        # if hole>1 and format=='NHWC':
        #     conv=tf.nn.atrous_conv2d(inpOp, kernel, hole, padding=padType)
        # else:
        #     conv = tf.nn.conv2d(input=inpOp,filter=kernel,strides= strides, padding=padType,
        #                      data_format=format,dilations=dilations,name=name)

        if format == 'NHWC':
            if hole > 1:
                conv = tf.nn.atrous_conv2d(inpOp, kernel, hole, padding=padType)
            else:
                conv = tf.nn.conv2d(input=inpOp, filter=kernel, strides=strides, padding=padType,
                                data_format=format, name=name)

        if format == 'NCHW':
            conv = tf.nn.conv2d(input=inpOp, filter=kernel, strides=strides, padding=padType,
                                    data_format=format,dilations=dilations, name=name)


        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=type),
                             trainable=True, name='biases')
        #bias = tf.reshape(tf.nn.bias_add(conv, biases,
        #                                data_format=format),
        #                  conv.get_shape(),name='reshaping')
        bias =tf.nn.bias_add(conv, biases,
                                         data_format=format)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1


def inference(images,type,format):#//This is Balinet Topology
    global conv_counter
    conv_counter=1
    with tf.variable_scope('WeightSharing') as scope:
        print('Inference Graph type:', type)
        conv1 = _conv (images, 1, 32, 5, 5, 1, 1, 'VALID',1,type,format) #inpOp, nIn, nOut, kH, kW, dH, dW, padType #-131
        conv2 = _conv(conv1, 32, 48, 3, 3, 1, 1, 'VALID',2,type,format) #-65
        conv3 = _conv (conv2,  48, 56, 3, 3, 1, 1, 'VALID',4,type,format)#32
        conv4 = _conv (conv3,  56, 64, 3, 3, 1, 1, 'VALID',8,type,format)#
        conv5 = _conv (conv4,  64, 128, 2, 2, 1, 1, 'VALID',1,type,format)
        conv6 = _conv (conv5,  128, 256, 1, 1, 1, 1, 'VALID',1,type,format)
        heatMap = _conv (conv6,  256, 6, 1, 1, 1, 1, 'VALID',1,type,format)

        Y_hat = tf.nn.softmax(heatMap)  # softmax apply to logits
        predicted_label = tf.argmax(Y_hat, 3)
    # affn1 = _affine(resh1, 1024 * 6 * 6, 3072)
    # affn2 = _affine(affn1, 3072, 4096)
    # affn3 = _affine(affn2, 4096, 1000)
    #
    return predicted_label

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)
#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#We need only the generator to estimate the inference time:
def generatorPix2Pix(cond_img, type, format,TrainingMode=False):
    ClassCount = 1
    with tf.variable_scope('gen/generator'):
        with slim.arg_scope([slim.batch_norm],is_training=TrainingMode):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.leaky_relu,#tf.nn.relu,#tf.nn.relu,
                                kernel_size=[4, 4],
                                stride=2):
                enc1 = slim.conv2d(cond_img, 64, scope='enc1')
                enc2 = slim.batch_norm(slim.conv2d(enc1, 128, scope='enc2'))
                enc3 = slim.batch_norm(slim.conv2d(enc2, 256, scope='enc3'))
                enc4 = slim.batch_norm(slim.conv2d(enc3, 512, scope='enc4'))
                enc5 = slim.batch_norm(slim.conv2d(enc4, 512, scope='enc5'))
                enc6 = slim.batch_norm(slim.conv2d(enc5, 512, scope='enc6'))
                enc7 = slim.batch_norm(slim.conv2d(enc6, 512, scope='enc7'))
                enc8 = slim.conv2d(enc7, 512, kernel_size=[2, 2], scope='enc8')

            with slim.arg_scope([slim.conv2d_transpose],
                                activation_fn=tf.nn.leaky_relu,#tf.nn.relu,#leaky_relu
                                kernel_size=[4, 4],
                                stride=2):
                dec1 = slim.batch_norm(slim.conv2d_transpose(enc8, 512, kernel_size=[2, 2], scope='dec1'))
                dec1 = tf.concat([dec1, enc7], 3, name='cat1')
                dec2 = slim.batch_norm(slim.conv2d_transpose(dec1, 512, scope='dec2'))
                dec2 = tf.concat([dec2, enc6], 3, name='cat2')
                dec3 = slim.batch_norm(slim.conv2d_transpose(dec2, 512, scope='dec3'))
                dec3 = tf.concat([dec3, enc5], 3, name='cat3')
                dec4 = slim.batch_norm(slim.conv2d_transpose(dec3, 512, scope='dec4'))
                dec4 = tf.concat([dec4, enc4], 3, name='cat4')
                dec5 = slim.batch_norm(slim.conv2d_transpose(dec4, 256, scope='dec5'))
                dec5 = tf.concat([dec5, enc3], 3, name='cat5')
                dec6 = slim.batch_norm(slim.conv2d_transpose(dec5, 128, scope='dec6'))
                dec6 = tf.concat([dec6, enc2], 3, name='cat6')
                dec7 = slim.batch_norm(slim.conv2d_transpose(dec6, 64, scope='dec7'))
                dec7 = tf.concat([dec7, enc1], 3, name='cat7')
                # dec8 = slim.conv2d_transpose(dec7, 3, activation_fn=tf.nn.tanh)
                Out = slim.batch_norm(slim.conv2d_transpose(dec7, ClassCount, scope='dec8'))
                # dec3 = slim.batch_norm(slim.conv2d_transpose(dec2, 64, scope='dec3'))
                # dec3 = tf.concat([dec3, enc1], 3, name='cat3')
                # Out = slim.batch_norm(slim.conv2d_transpose(dec3, ClassCount, scope='dec4'))

            return Out

# def generatorPix2Pix(cond_img, type, format):
#     ClassCount = 1
#     with tf.variable_scope('gen/generator'):
#         with slim.arg_scope([slim.conv2d],
#                             activation_fn=lrelu,
#                             kernel_size=[5, 5],
#                             stride=2):
#             enc1 = slim.conv2d(cond_img, 64, scope='enc1')
#             enc2 = slim.batch_norm(slim.conv2d(enc1, 128, scope='enc2'))
#             enc3 = slim.batch_norm(slim.conv2d(enc2, 256, scope='enc3'))
#             enc4 = slim.batch_norm(slim.conv2d(enc3, 512, scope='enc4'))
#             enc5 = slim.batch_norm(slim.conv2d(enc4, 512, scope='enc5'))
#             enc6 = slim.batch_norm(slim.conv2d(enc5, 512, scope='enc6'))
#             enc7 = slim.batch_norm(slim.conv2d(enc6, 512, scope='enc7'))
#             enc8 = slim.batch_norm(slim.conv2d(enc7, 512, kernel_size=[2, 2], scope='enc8'))
#
#         with slim.arg_scope([slim.conv2d_transpose],
#                             activation_fn=tf.nn.relu,
#                             kernel_size=[4, 4],
#                             stride=2):
#             dec1 = slim.batch_norm(slim.conv2d_transpose(enc8, 512, kernel_size=[2, 2], scope='dec1'))
#             dec1 = tf.concat([dec1, enc7], 3, name='cat1')
#             dec2 = slim.batch_norm(slim.conv2d_transpose(dec1, 512, scope='dec2'))
#             dec2 = tf.concat([dec2, enc6], 3, name='cat2')
#             dec3 = slim.batch_norm(slim.conv2d_transpose(dec2, 512, scope='dec3'))
#             dec3 = tf.concat([dec3, enc5], 3, name='cat3')
#             dec4 = slim.batch_norm(slim.conv2d_transpose(dec3, 512, scope='dec4'))
#             dec4 = tf.concat([dec4, enc4], 3, name='cat4')
#             dec5 = slim.batch_norm(slim.conv2d_transpose(dec4, 256, scope='dec5'))
#             dec5 = tf.concat([dec5, enc3], 3, name='cat5')
#             dec6 = slim.batch_norm(slim.conv2d_transpose(dec5, 128, scope='dec6'))
#             dec6 = tf.concat([dec6, enc2], 3, name='cat6')
#             dec7 = slim.batch_norm(slim.conv2d_transpose(dec6, 64, scope='dec7'))
#             dec7 = tf.concat([dec7, enc1], 3, name='cat7')
#             # dec8 = slim.conv2d_transpose(dec7, 3, activation_fn=tf.nn.tanh)
#             Out = slim.batch_norm(slim.conv2d_transpose(dec7, ClassCount, scope='dec8'))
#             # dec3 = slim.batch_norm(slim.conv2d_transpose(dec2, 64, scope='dec3'))
#             # dec3 = tf.concat([dec3, enc1], 3, name='cat3')
#             # Out = slim.batch_norm(slim.conv2d_transpose(dec3, ClassCount, scope='dec4'))
#
#         return Out



def FreezetoFile(ckpt_path,meta_path,output_frozen_graph_name):

    input_graph_path = meta_path
    checkpoint_path = ckpt_path
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output"
    restore_op_name = ""
    filename_tensor_name = ""

    clear_devices = True
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

def loadFrozenGraph(FGraphPAth):
    with gfile.FastGFile(FGraphPAth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def Run_Pix2Pix_graph_def(dummy_input,inferenceCallFctn,savePath,num_loops,type,format,TimeLineFullNamePath,ChckPointFP32=0):
  #Create the Pix2Pix Graph for inference
  # Call the initialization
  #Run A few time to get some stat
  #Save the chkpt
  tf.reset_default_graph()
  g = tf.Graph()
  dynamic_shape = list(dummy_input.shape)

  with g.as_default():

        if format=='NHWC':
            images = tf.placeholder(
                dtype=type, shape=dynamic_shape, name="input")
        else:
            images = tf.placeholder(
                dtype=type, shape=dynamic_shape, name="input")
        last_layer = inferenceCallFctn(images,type,format)
        last_layer = tf.identity(last_layer, name="output")
        saver = tf.train.Saver()
        # Start running operations on the Graph.
        sess = tf.Session()
        # ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        # # ckpt.model_checkpoint_path.
        # print(ckpt.model_checkpoint_path)
        if ChckPointFP32:
            print('Restoring Checkpoint from training')
            if type=="float32":
                saver.restore(sess, ChckPointFP32) ## Restore Checkpoint as is
            else: ## We need to cast manually each variable:
                print('float 32 bit variables from  training   are converted to float 16 bit')
                previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(ChckPointFP32)]
                # sess.run(tf.global_variables_initializer())
                restore_map = {}
                for variable in tf.global_variables():
                    if variable.op.name in previous_variables:
                        var = tf.contrib.framework.load_variable(
                            ChckPointFP32, variable.op.name)
                        if (var.dtype == np.float32):
                            tf.add_to_collection('assignOps', variable.assign(
                                tf.cast(var, tf.float16)))
                        else:
                            tf.add_to_collection('assignOps', variable.assign(var))
                sess.run(tf.get_collection('assignOps'))


        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(20):
            _val = sess.run(last_layer, {images: dummy_input})
        tf.logging.info("Warmup done. Starting real timing")
        num_iters = 5
        timings=[]
        for i in range(num_loops):
            if i < num_loops - 1:
                tstart = time.time()
                for k in range(num_iters):
                  _val = sess.run(last_layer, {images: dummy_input})
                timings.append((time.time() - tstart) / float(num_iters))
                print("Tensorflow Run iter ", i, " ", timings[-1])
          # Create the Timeline object, and write it to a json file
            else: #(i=num_loops-1 : #Last Loop only
                _val = sess.run(last_layer, {images: dummy_input}, options=options, run_metadata=run_metadata)
                _val = sess.run(last_layer, {images: dummy_input}, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(TimeLineFullNamePath, 'w') as f:
            f.write(chrome_trace)
        tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), savePath,
                             type.name+'.pbtxt', as_text=True)


        meta_path = os.path.join(savePath, type.name+'.pbtxt')
        ckpt_path = saver.save(sess, os.path.join(savePath,type.name+'-ckpt'))
        print("Model saved in path: %s" % ckpt_path)
  return timings,ckpt_path,meta_path


def getFrozenGraph(FrozenPath):

  with gfile.FastGFile(FrozenPath,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  return graph_def

def printStats(graphName,timings,batch_size):

    if timings is None:
        return
    times=np.array(timings)
    speeds=batch_size / times
    avgTime=np.mean(timings)
    avgSpeed=batch_size/avgTime
    stdTime=np.std(timings)
    stdSpeed=np.std(speeds)
    print("images/s : %.1f +/- %.1f, s/batch: %.5f +/- %.5f, s/frame: %.5f +/- %.5f"%(avgSpeed,stdSpeed,avgTime,stdTime,avgTime/batch_size,stdTime/batch_size))
    print("RES, %s, %s, %.2f, %.2f, %.5f, %.5f"%(graphName,batch_size,avgSpeed,stdSpeed,avgTime,stdTime))
    print("RES, %s, s/frame %.5f" % (graphName, avgTime/batch_size))

#
def getTRTGraph(batch_size,workspace_size,FrozenPath,type):
  if type.name=='float32':
      precision_mode='FP32'
  elif type.name=='float16':
    precision_mode = 'FP16'
  else:
      assert("unsupported precision_mode")

  trt_graph = trt.create_inference_graph(getFrozenGraph(FrozenPath), ["output"],
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode=precision_mode)  # Get optimized graph
  newPath=os.path.join(os.path.dirname(FrozenPath),'_TRT'+precision_mode+'.pb')
  with gfile.FastGFile(newPath,'wb') as f:
    f.write(trt_graph.SerializeToString())
  return trt_graph


def timeGraph(gdef, batch_size=128, num_loops=100, dummy_input=None,TimeLineFullNamePath='testLog.json'):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
  tf.reset_default_graph()
  g = tf.Graph()
  if dummy_input is None:
    dummy_input = np.random.random_sample((batch_size, 100, 100, 1)).astype(np.float32)
  outlist = []
  with g.as_default():

    inp, out = tf.import_graph_def(
      graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]

    outlist.append(out)


  timings = []

  with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    tf.logging.info("Starting Warmup cycle")



    for i in range(20):
     valt = sess.run(out,{inp: dummy_input})
    tf.logging.info("Warmup done. Starting real timing")
    num_iters = 3
    for i in range(num_loops):
      tstart = time.time()
      for k in range(num_iters):
        val = sess.run(out, {inp: dummy_input},options=run_options,run_metadata=run_metadata)
      timings.append((time.time() - tstart) / float(num_iters))
      print("iter ", i, " ", timings[-1])
      if i == num_loops - 1:  # Last Loop only
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(TimeLineFullNamePath, 'w') as f:
         f.write(chrome_trace)
    comp = sess.run(tf.reduce_all(tf.equal(val[0], valt[0])))
    print("Comparison=", comp)
    sess.close()
    tf.logging.info("Timing loop done!")
    return timings, comp, val[0], None

def makeDirsRes(netName):
    savePath = 'GraphFiles_'+netName
    if not os.path.exists(savePath):
      os.makedirs(savePath)
    else:
      shutil.rmtree(savePath)  # removes all the subdirectories!
      os.makedirs(savePath)

    timelinesPath = 'Timelines_'+netName
    if not os.path.exists(timelinesPath):
      os.makedirs(timelinesPath)
    else:
      shutil.rmtree(timelinesPath)  # removes all the subdirectories!
      os.makedirs(timelinesPath)
    return savePath,timelinesPath

def doNetTimingAnalysis(netName,Tensorformat,inferenceCallFctn,cmdArg,ChckPointFP32=0):
    print('#############################################')
    print('Timing Analysis for: ', netName)
    print('#############################################')
    savePath, timelinesPath = makeDirsRes(netName+'_'+Tensorformat)



    if Tensorformat=='NHWC':
        dummy_input = np.random.random_sample((cmdArg.batch_size,512,512,1))
    else:
        dummy_input = np.random.random_sample((cmdArg.batch_size, 1,512, 512))
    # Native  inference (Within TF)
    if cmdArg.TF_nativeFP32:
        type = tf.float32
        expName = 'TF_native_' + Tensorformat + '_' + type.name
        print('Now running:' + expName)
        timings, ckpt_path, meta_path = Run_Pix2Pix_graph_def(dummy_input,inferenceCallFctn, savePath, cmdArg.num_loops, type,
                                                              Tensorformat, os.path.join(timelinesPath, expName) + '.json',ChckPointFP32)
        printStats(expName, timings, f.batch_size)
        # Let freeze the Graph for  inference
        FrozenPath = os.path.join(savePath, 'Frozen_FP32.pb')
        FreezetoFile(ckpt_path, meta_path, FrozenPath)

        expName = 'TF_Frozen_' + Tensorformat + '_' + type.name
        print('================================\n')
        print('Now running:' + expName)
        timings, comp, valnative, mdstats = timeGraph(getFrozenGraph(FrozenPath), cmdArg.batch_size,
                                                      cmdArg.num_loops, dummy_input,
                                                      os.path.join(timelinesPath, expName) + '.json')

        printStats(expName, timings, cmdArg.batch_size)

    if cmdArg.TF_nativeFP16:
        dummy_input = dummy_input.astype(float)
        type = tf.float16
        expName = 'TF_native_' + Tensorformat + '_' + type.name
        print('================================\n')
        print('Now running:' + expName)
        timings, ckpt_path, meta_path = Run_Pix2Pix_graph_def(dummy_input, inferenceCallFctn, savePath,cmdArg.num_loops, type,
                                                              Tensorformat,os.path.join(timelinesPath, expName) + '.json')
        printStats(expName, timings, cmdArg.batch_size)

    # Start Tensor RT
    if isTRT:

        wsize = f.workspace_size << 20

        for type in [tf.float32, tf.float16]:
            expName = 'TRT-' + Tensorformat + '_' + type.name
            print('================================\n')
            print('Now running:' + expName)
            timings, comp, valfp32, mdstats = timeGraph(getTRTGraph(cmdArg.batch_size, wsize, FrozenPath, type), cmdArg.batch_size,
                                                cmdArg.num_loops, dummy_input,
                                                os.path.join(timelinesPath, expName) + '.json')
            printStats(expName, timings, f.batch_size)




if "__main__" in __name__:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    P=argparse.ArgumentParser(prog="test")

    f,unparsed=P.parse_known_args()

    f.batch_size=4
    f.num_loops=5
    f.workspace_size=2024

    f.TF_nativeFP32=True
    f.TF_nativeFP16 = True
    f.FP32=True #Run Tensor RT
    f.FP16=True #Run Tensor RT

    #The Script will :
    # 1 built the graph and run it in TF
    # 2 Freeze it and run it in TF
    # 3 convert it to TensorRT and run it
    #The analysis is repeated for all the relevant combinations FP16,FP32,TensorFormat can be (NWHC,NCWH)
    #The analysis is performed on 2 different networks BaliNet and Pix2Pix
    #All the timeline results are stored in separate folders

    # Balinet case
    inferenceCallFctn = inference


    TsrFrmtLst=['NHWC']
    if float(tf.__version__[0:-2]) > 1.4:
        TsrFrmtLst.append('NCHW')
    #Use Balinet Check point
    Balinet_Chkpt = "BaliNetChkpt/model.ckpt-7500"
    for Tensorformat in TsrFrmtLst:
      doNetTimingAnalysis('BaliNet', Tensorformat, inferenceCallFctn, f,Balinet_Chkpt)

    #pix2pix case
    #Use Pix2pix Check point
    # Pix2Pix_Chkpt="/nfs/home/laurent/TensorPack/tensorpack/examples/GAN/train_log/Image2Image_Unet_Pix2Pix_ST-NISI-Big-PixelSize_GAN/model-39840"
    Pix2Pix_Chkpt = "Pix2PixChkpt/model.ckpt"
    inferenceCallFctn = generatorPix2Pix
    Tensorformat='NHWC'
    doNetTimingAnalysis('Pix2Pix',Tensorformat,inferenceCallFctn,f,Pix2Pix_Chkpt)



sys.exit(0)
