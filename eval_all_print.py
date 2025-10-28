import os
import json
import subprocess
# model_folders=["0331_20D_mid_2k_32_4_4", "0331_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs256.32bi_best_vlm2vecqwen2b_mid_2k_64_8_8"]
model_folders=["0410_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0410_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy", "0411_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"]
model_folders=["0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0415_InstuctionP_odibn_sdibn_20D_HNrand1PS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy",
               "0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy", "0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.1024bi_30.2030_30P.10.5_70.2070_qwen2b_2k_dy", 
               "0415_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0416_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy",
               "0416_InstuctionP_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0416_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy/checkpoint-2000",
               "0415_copy_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy","0422_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy",
               "0420_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy",
               "0420_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy/merged/", "0421_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen7b_2k_dy",
               "0420_InstuctionP_odibn_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_vlm2vecqwen2b_2k_dy/"]

model_folders=["0420_InstuctionP_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy", "0418_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs32.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0417_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs128.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0421_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0420_InstuctionP_rdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_64k_dy", "0420_InstuctionP_rdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy", "0422_InstuctionP_rdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy", "0423_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b.qwen7b_2k_dy", "0423_InstuctionP_rdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0415_InstuctionP_odibn_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen7b.qwen2b_2k_dy"]
model_folders=["0424_InstuctionP_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen7b_2k_dy", "0424_InstuctionP_rdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_train1_2k_dy/with_inst", "0425_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy",
               "0425_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen7b_2k_dy", "0425_InstuctionP_odibn8_sdibn_20D_HNPS_Metis_bs1024.8bi_30.130_30P.10.5_70.170_qwen7b_2k_dy", "0425_InstuctionP_odibn8_sdibn_20D_HNPS_Metis_bs1024.8bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", 
               "0425_InstuctionP_odibn16_sdibn_20D_HNPS_Metis_bs1024.16bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0425_InstuctionP_odibn16_sdibn_20D_HNPS_Metis_bs1024.16bi_30.130_30P.10.5_70.170_qwen7b_2k_dy",
                "0429_InstuctionP_odibn_sdibn1024_20D_HNNone_Metis_bs512.1024bi_30.2030_30P.10.5_70.2070_qwen2b_2k_dy", "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy",
                "0429_InstuctionP_odibn128_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy"]


model_folders=["0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs64.32bi_30.130_30P.10.5_70.170_qwen2b_32k_dy", "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_64k_dy", "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_16k_dy", "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0429_InstuctionP_odibn8_sdibn_20D_HNNone_Metis_bs512.8bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"]+[
    "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0429_InstuctionP_odibn_sdibn1024_20D_HNNone_Metis_bs512.1024bi_30.2030_30P.10.5_70.2070_qwen2b_2k_dy", "0429_InstuctionP_odibn128_sdibn_20D_HNNone_Metis_bs512.128bi_30.330_30P.10.5_70.370_qwen2b_2k_dy",
    "0429_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0429_InstuctionP_odibn32_sdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"
]

model_folders=["0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_0.100_0.30P.10.5_0.100_qwen2b_2k_dy"]
model_folders=["0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs128.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0503_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy", "0504_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs32.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"]
model_folders=["0505_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_0.30_0.10P.10.5_0.70_qwen2b_2k_dy"]
model_folders=["0505_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-200"]
model_folders=["0505_InstuctionP_odibn32_sdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-600"]
model_folders=["internvl37b_checkpoint-600"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-600"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b0.5k_2k_dy"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy/checkpoint-1000"]
model_folders=["qwen25_checkpoint-1000"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b3.5k_2k_dy", "0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b0.5k_2k_dy"]
model_folders=["0508_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.2b_2k_dy"]
model_folders=["0507_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs1024.32bi_30.130_30P.10.5_70.170_internvl3.8b_2k_dy"]
model_folders=["0510_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_qwen7b.qwen2b_2k_dy"]
model_folders=["0511_InstuctionP_odibn32_sdibn_20D_HNPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen257b_2k_dy_modified/"]
model_folders=["0513_InstuctionP_rdibn_20D_HNIPS_Metis_bs1024.32bi_30.130_30P.10.5_70.170_qwen2b_2k_dy"]
model_folders=["0728_InstuctionP_odibn32_sdibn_20D_HNNone_Metis_bs512.32bi_30.130_30P.10.5_70.170_clip.qwen2b_2k_dy"]
# Directory containing all model folders
base_directory = "./MMEB-evaloutputs"

# Iterate over each model folder
for model in model_folders:
    model_path = os.path.join(base_directory, model)
    accuracies = [model]  # Start with the model name
    
    if ("5CD" in model):
        datasets = ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"]
    elif ("6VQA" in model):
        datasets = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz"]
    elif ("8D" in model):
        datasets = ["VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ","Wiki-SS-NQ", "OVEN",  "EDIS", ]
    else:
        datasets = ["VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ","Wiki-SS-NQ", "OVEN",  "EDIS",
                    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
                    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz"]

    datasets = ["VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ","Wiki-SS-NQ", "OVEN",  "EDIS",
                    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
                    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz", "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]
        # assert False, "Not an accepted model"

    for dataset in datasets:
        dataset_file = f"{dataset}_score.json"
        dataset_path = os.path.join(model_path, dataset_file)
        
        # Check if the dataset file exists and extract accuracy
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as file:
                data = json.load(file)
                acc = data.get("acc", "N/A")  # Get accuracy, default to "N/A" if missing
                if acc != "N/A":
                    acc = round(acc * 100, 2)  # Multiply by 100 and round to 2 decimal places
                accuracies.append(str(acc))  # Convert to string for uniform output
        else:
            accuracies.append("N/A")  # File not found or missing value``
    
    # Print the model name followed by all accuracies, space-separated
    print(" ".join(accuracies))
