export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
python src/train.py --loaded --fixed --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN
python src/train.py --loaded --fixed --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN_small
python src/train.py --loaded --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN
python src/train.py --loaded --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN_small
python src/train.py --fixed --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN
python src/train.py --fixed --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN_small
python src/train.py --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN
python src/train.py --dataset_path=/nfs/nas-7.1/yelin/AI2021Spring/Face-Mask-Detector/data/MFN_small