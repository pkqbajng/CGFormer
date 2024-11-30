set -e
exeFunc(){
    cd mobilestereonet
    baseline=$1
    num_seq=$2
    data_path=$3
    CUDA_VISIBLE_DEVICES=1 python prediction.py --datapath $data_path/sequences/$num_seq \
    --testlist ./filenames/$num_seq.txt --num_seq $num_seq --loadckpt ./MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti \
    --model MSNet3D --savepath $data_path/depth --baseline $baseline
    cd ..
}

data_path='data/semantickitti'
for i in {00..02}
do
    exeFunc 388.1823 $i $data_path
done

for i in {03..03}
do
    exeFunc 389.6304 $i $data_path
done

for i in {04..12}
do
    exeFunc 381.8293 $i $data_path
done

for i in {13..21}
do
    exeFunc 388.1823 $i $data_path
done
