# bash
for ((i=0;i<=2;i++))
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py hmdbucf RGB \
     --arch resnet50 --num_segments 5 \
     --num-spatial-group 4 --entropy-min 0.15 --entropy-mean-max 0.15 --cross-relation 0.5 \
     --mix-layers 1 2 3 --mix-alpha 0.1 --mix-prob 0.5 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 32 -j 16 --dropout 0.3 \
     --eval-freq=1 --print-freq 5 --suffix '4gpu_bz32_mix123_group4_dropout3' --seed ${i} 
done

