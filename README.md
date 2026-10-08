# nanoWxRisk

把"时序/频域编码器"那条研究线换成**视频编码器类比**，用于气象灾害风险指数。

- **输入**：起始条件（analysis 场）+ 一组超算集合预报成员（IFS ENS，50 成员）
- **输出**：给定地理网格上、给定提前期的指定灾害风险指数（事件发生概率 / 指数分布）
- **直觉**：大气演变 = 二维特征在时间上的运动 → I 帧（初始场）+ P 帧（各成员预报场）；
  成员之间的分歧 = 编码器的"模式竞争"，用 RD 式投票决定哪个特征胜出，胜出模式的
  熵/分歧度直接给出风险指数的分布。

设计与实验协议见 [DESIGN.md](DESIGN.md)。研究日志见 `notebooks/`。

## 数据（公开、免认证）

WeatherBench-2 的 GCS 公共桶 `gs://weatherbench2`：

| 用途 | 路径 | 形状 |
| --- | --- | --- |
| 集合预报 | `datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr` | time 3652 × number 50 × lead 61 × lon 64 × lat 32 |
| 真值 | `datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr` | time 93544 × lon 64 × lat 32 |

64×32 网格（5.625°）是刻意选的"nano"规模：全球 2048 个格点，单机可跑完整流程。
分辨率阶梯 64x32 → 240x121 → 1440x721 都在同一个桶里，结论站住了再升分辨率。

## 快速开始

```shell
uv sync
.venv/bin/python scripts/01_fetch_subset.py --years 2018 2021 --lead-hours 72
.venv/bin/python scripts/02_baseline_vote.py
```
