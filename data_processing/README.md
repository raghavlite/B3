## Clustering

This step requires the **N×N teacher-ranked matrix** to be available at the `MODEL_BASE_PATH`.

```bash
python spectral_clustering4_ibn.py --dataset MSCOCO_i2t --negs hn --nmax 100 --batch_size 32 --K 5
```

---

## Adding Instructions

This step uses the **clustered outputs** generated from the previous stage.

```bash
python modify_posnegtext_instruction2.py
```

Multiple clustering files with different parameters have already been created and provided in **`B3/MMEB-train2`**.  
For example,  
`bs32bi_30.130_qwen2b` means:
- Teacher model: `vlm2vecqwen2b`
- Cluster size: $K = 32$
- Parameters: $p = 30$, $m = 100$
