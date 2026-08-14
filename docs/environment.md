# 实验环境

|组件|实验版本|
|---|---|
|设备|4 × K500SM_AI（实验限制使用单卡）|
|单卡显存|65520 MiB|
|Python|3.10.12|
|vLLM|0.11.0+das.opt1.alpha.dtk25042|
|PyTorch|2.5.1+das.opt1.dtk25042|
|Transformers|4.57.3|
|Triton|3.1.0+das.opt1.dtk25042|
|精度|BF16|

实验运行在组织提供的海光 DTK 定制容器中。公开仓库不提供内部镜像地址，也不
提供模型权重。复现者应使用与自己硬件和驱动匹配的 PyTorch/vLLM 环境。

原始环境核验输出位于 `verification/environment_check.txt`。

