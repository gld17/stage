# Plan Quiz — flexet-phase1-rename-inference

## Q1
关于训练相关环境变量的处理，以下哪项正确？

A. 所有 `STAGE_*` 环境变量均保留，仅将前缀改为 `FLEXET_*`
B. `FLEXET_OPTIMIZED`、`FLEXET_LEGACY_ATTR`、`FLEXET_MERGE_COMMS` 改为 `FLEXET_*`，`STAGE_MICROBATCH_OPTIMIZE` 直接删除
C. `STAGE_MICROBATCH_OPTIMIZE` 改为 `FLEXET_MICROBATCH_OPTIMIZE`，因为它在推理时也可能有用
D. 所有环境变量均直接删除，不保留任何 `FLEXET_*` 前缀

## Q2
关于 Pipeline Parallel 中 "stage" 一词的处理，以下哪项正确？

A. 所有包含 "stage" 的文件名、变量名、函数名均应替换为 "flexet"
B. 仅产品名和目录名中的 "stage" 替换为 "flexet"，`_stage_for_layer()` 等 pipeline stage 术语保留不变
C. 所有 "stage" 均保留，不作任何替换
D. `flexet/` 目录重命名为 `flexet/`，但内部文件中的 `stage` 变量名全部替换
