iters=320000
config="configs/LMSeg/semantic_4datasets/lmseg_4datasets__clip50__dataset-aware-aug__bs16.yaml"
outdir="training_dir/lmseg_4datasets__clip50__dataset-aware-aug__bs16__320k"

empty_cache_step=-1


mkdir -p ${outdir}
cp $0 ${outdir}/


for i in {0..1}
do
    python train_lmseg_net.py --num-gpus 8 --resume \
        --config-file ${config} OUTPUT_DIR ${outdir} SOLVER.MAX_ITER ${iters} \
        LMSEG.EMPTY_CACHE_STEP ${empty_cache_step}
done
