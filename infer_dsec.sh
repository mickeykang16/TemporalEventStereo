logdir=exps/debug

python3 infer_dsec.py --config config/dsec_infer.yaml\
               --savemodel ${logdir}

curr_dir=$PWD
cd $logdir
zip -r test.zip test
cd $curr_dir
echo Gpu number is ${GPUNUM}!
echo Done!