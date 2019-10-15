
# The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
# Object detection script writes output to a file inside a directory. We make sure that this directory exists.
#  The output directory is the first argument of the bash script
mkdir -p $OUTPUT_FILE
ROIFILE=$OUTPUT_FILE/ROIs.txt
OVIDEO=$OUTPUT_FILE/output.mp4

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNetCaffe.aocx 
fi

if [ "$FP_MODEL" = "FP32" ]; then
    config_file="conf_fp32.txt"
else
    config_file="conf_fp16.txt"
fi

# Running the object detection code
SAMPLEPATH=$PBS_O_WORKDIR
./tutorial1 -i /data/reference-sample-data/object-detection-python/cars_1900.mp4 \
            -m models/mobilenet-ssd/$FP_MODEL/mobilenet-ssd.xml \
            -d $DEVICE \
            -o $OUTPUT_FILE\
            -fr 3000 

# Converting the text output to a video
./ROI_writer -i /data/reference-sample-data/object-detection-python/cars_1900.mp4 \
             -o $OUTPUT_FILE \
             -ROIfile $ROIFILE \
             -l pascal_voc_classes.txt \
             -r 2.0 # output in half res
