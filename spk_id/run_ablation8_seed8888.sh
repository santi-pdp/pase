#!/bin/bash

# VCTK

#sbatch --array=1-4%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL_SEED3333
#sbatch --array=5-8%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-CHUNK_SEED3333
#sbatch --array=9-12%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-SPC_SEED3333
#sbatch --array=13-16%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-CMI_SEED3333
#sbatch --array=17-20%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-MI_SEED3333
#sbatch --array=21-24%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-PROSO_SEED3333
#sbatch --array=25-28%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-LPS_SEED3333
#sbatch --array=29-32%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-MFCC_SEED3333
#sbatch --array=33-36%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL2_SEED3333

# INTERFACE

sbatch --array=1-4%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL_INTERFACE_SEED8888
sbatch --array=5-8%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-CHUNK_INTERFACE_SEED8888
sbatch --array=9-12%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-SPC_INTERFACE_SEED8888
sbatch --array=13-16%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-CMI_INTERFACE_SEED8888
sbatch --array=17-20%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-MI_INTERFACE_SEED8888
sbatch --array=21-24%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-PROSO_INTERFACE_SEED8888
sbatch --array=25-28%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-LPS_INTERFACE_SEED8888
sbatch --array=29-32%1 run_UE_point.sh ablation8_U.guia ablation8_variation_seeds/ALL-MFCC_INTERFACE_SEED8888

