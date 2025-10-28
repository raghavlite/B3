#!/bin/bash

partitions=("wiseman" "nlplab" "nlplab-core")
partitions=("wiseman" "nlplab" "nlplab-core")
partitions=("nlplab-core" "nlplab" "wiseman" "bhuwan")
# partitions=("compsci-gpu")
# List of model names
# models=("VLM2Vec-Qwen2VL-2B")

# models=("0322_20D_vlm2vec128qwen2b_mid_2k_32_3_3" "0320_20D_vlm2vecqwen2b_mid_2k_16_3_3")
# models=("0322_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs32bi_best_vlm2vecqwen2b_mid_2k_16_3_3")
# models=("TIGER-Lab/VLM2Vec-Qwen2VL-2B")
# models=("0322_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs128bi_best_vlm2vecqwen2b_mid_2k_32_4_4")
# models=("0326_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs32bi_best_vlm2vecfull_8k_16_2_2")
# models=("0331_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs128.32bi_best_vlm2vecqwen2b_mid_2k_32_4_4")
# models=("VLM2Vec-Qwen2VL-7B")
# models=("0331_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs128.32bi_best_vlm2vecqwen2b_mid_2k_32_4_4")
# models=("0305_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs128bi_best_vlm2vecqwen2b_mid_2k_32_4_4")
# models=("0331_20D_mid_2k_32_4_4")
# models=("0331_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs256.32bi_best_vlm2vecqwen2b_mid_2k_64_8_8")
models=("0410_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy" "0410_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0411_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy")
models=("0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0415_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy")
models=("0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.1024bi_30.2030_30P.10.5_70.2070_qwen2b_2k_dy")
models=("0415_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0416_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0416_InstuctionP_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0416_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy/checkpoint-2000")
models=("0415_copy_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0422_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy")
models=("0420_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy/merged")
models=("0420_InstuctionP_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy" "0418_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs32.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0417_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs128.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0421_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0420_InstuctionP_rdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_64k_dy" "0420_InstuctionP_rdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy" "0422_InstuctionP_rdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy")
models=("0423_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b.qwen7b_2k_dy" "0423_InstuctionP_rdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy" "0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen7b.qwen2b_2k_dy")
models=("0424_InstuctionP_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen7b_2k_dy")
models=("0424_InstuctionP_rdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_train1_2k_dy")
models=("0425_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0425_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen7b_2k_dy")
models=("0425_InstuctionP_odibn8_sdibn_20D_HNPS_Metis_bs1024.8bi_30.130_30P.10.5_70.170_qwen7b_2k_dy")
models=("0425_InstuctionP_odibn8_sdibn_20D_HNPS_Metis_bs1024.8bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0425_InstuctionP_odibn16_sdibn_20D_HNPS_Metis_bs1024.16bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0429_InstuctionP_odibn128_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy")
models=("0425_InstuctionP_odibn16_sdibn_20D_HNPS_Metis_bs1024.16bi_30.130_30P.10.5_70.170_qwen7b_2k_dy")
models=("0429_InstuctionP_odibn_sdibn1024_20D_HNNone_Metis_bs512.1024bi_30.2030_30P.10.5_70.2070_qwen2b_2k_dy" "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy" "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_64k_dy" "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy")
models=("0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy" "0429_InstuctionP_odibn8_sdibn_20D_HNNone_Metis_bs512.8bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0429_InstuctionP_odibn32_sdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy")
models=("0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy" "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy" "0504_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0505_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy")
models=("0507_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.8b_2k_dy/checkpoint-200" "0505_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-200")
models=("0505_InstuctionP_odibn32_sdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("internvl37b_checkpoint-600")
models=("qwen25_checkpoint-1000")
models=("0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-1000")
models=("0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b3.5k_2k_dy")
models=("0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b0.5k_2k_dy")
models=("0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy")
models=("0507_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.8b_2k_dy")
models=("0510_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen7b.qwen2b_2k_dy")
models=("0511_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen257b_2k_dy_modified/")
models=("0513_InstuctionP_rdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy")
models=("0728_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_clip.qwen2b_2k_dy")
counter=0
# for i in {0..14}; do
# for i in 15 18; do
# for i in {15..19}; do
# for i in {0..6}; do
partitions=("wiseman")
for model in "${models[@]}"; do
    partition=${partitions[$counter % ${#partitions[@]}]}  # Cycle through partitions
    if [[ "$model" == *"VLM2Vec"* ]]; then
        :
        # sbatch -A $partition --partition=$partition --array=${i}%1 --ntasks=1 --mem=100gb --output=./Nlogs_extra/0313_labelling10k/${model}/%03a.out ./eval_all.sh TIGER-Lab/${model} ./MMEB-evaloutputs/0313_labelling10k/${model}/ 
        # sbatch -A $partition --partition=$partition --array=${i}%1 --ntasks=1 --mem=350gb --output=./Nlogs_extra/0313_labelling10k/${model}/%03a.out ./eval_all.sh TIGER-Lab/${model} ./MMEB-evaloutputs/0313_labelling10k/${model}/
    else
        :
        # scp -r rt195@dcc-login-03.oit.duke.edu:/hpc/group/csdept/rt195/VLM2Vec-pro/MMEB-trainedmodels/${model} ./MMEB-trainedmodels/;
        sbatch --dependency=afterany:8480671 --partition=compsci-gpu --gres=gpu:a5000:1 --array=0-35%37 --ntasks=1 --mem=50gb --output=./Nlogs_extra/${model}/%03a.out ./eval_all.sh ./MMEB-trainedmodels/${model}/  ./MMEB-evaloutputs/${model}/
        # sbatch --partition=compsci-gpu --gres=gpu:a5000:1 --array=0-35%37 --ntasks=1 --mem=50gb --output=./Nlogs_extra/${model}/%03a.out ./eval_all.sh ${model}  ./MMEB-evaloutputs/${model}/
    fi
    ((counter++))  # Increment counter for partition switching
done
# done