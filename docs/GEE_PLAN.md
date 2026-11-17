# Google Earth Engine 展示方案（开发计划）

> **状态更新 2026-11-12：改用 plotly + Colab，GEE 不再是必需。** mentor 邮件只要求可交互可视化；学校设置也禁止发布 Earth Engine App。展示改为自包含的 Colab 笔记本 `notebooks/colab_county_viewer.ipynb`（数据压缩内嵌，约 6 MB，分享笔记本即分享全部；无需账号、Drive 或下载，只需 plotly）。由 `scripts/113_build_colab_notebook.py` 从 `scripts/110`、`112` 的导出生成。本文档下面的 GEE 部分保留作参考：`gee/app.js`（Code Editor 脚本）与 `scripts/111`（上传）可用，但 GEE 上的县×事件资产是旧版（得分被舍入到 4 位小数），**不要使用**；变量的定义（§3b）仍然有效。

状态：计划，未实现。日期 2026-11-10。依据：`docs/SCENARIO_REPORT.md`（展示建议）、`AGENTS.md`（评估规则）、Earth Engine 官方文档（上传表资产、项目注册）。

## 1. 目标与边界

用 Earth Engine (GEE) 作为**展示层**：地图上看 Texas 洪水线的 block-group 排序、事件回放、与真实索赔对照；SCS3 县级日风险作为第二图层。

- **不在 GEE 里跑模型。** 训练与推理（CNN、boosting、GEFS 特征）继续在 Python / Sol 上完成；GEE 只接收**已算好的结果表**。理由：模型依赖本地 `.npy`/zstd 数组和 torch，GEE 无法承载；展示只需要排序结果。
- 展示的是**排序**，不是绝对概率（AGENTS.md：绝对概率跨年不迁移）。图例用分位数 / top-k，不用 "x% 概率"。
- 必须在界面上标明模式：**forecast mode**（BG AUC 0.75，top-5% 捕获 20%）与 **observed-rain mode**（AUC 0.86，top-5% 43%）。不能混用同一套图例而不标注。
- 按 SCENARIO_REPORT：**用离散 block-group 点/面着色，不做平滑插值面**（点 AUC 0.78 vs 平滑 0.71–0.73）；插值只作背景。

## 2. 架构

```
Python (Mac / Sol)                      GCS bucket             Earth Engine                 前端
 scripts/110_gee_export.py  ──parquet→CSV/GeoJSON/SHP──►  gs://<bucket>/wxrisk/  ──upload table──►  projects/<proj>/assets/wxrisk/*  ──►  EE App / 自建网页
 (排名、事件表、BG 几何、县几何)                           (earthengine upload table)       (FeatureCollection)         (图层、时间滑块、图表)
```

三条路线，建议分两阶段：

| 阶段 | 做法 | 适用 | 代价 |
|---|---|---|---|
| A（先做） | Code Editor 脚本 + **Earth Engine Apps** 发布 | 内部/导师演示，最快 | UI 受限，发布与共享受 EE 权限设置约束 |
| B（需要对外/定制 UI） | 自建网页，用 `ee` JS 客户端 + 服务账号经后端代理取 tile（`getMapId`） | 公开网站、自定义图表 | 要维护后端与凭据；公开使用需确认项目用途（非商业/商业注册） |

先做 A，数据资产结构与 B 通用。

## 3. 展示内容（按优先级）

1. **事件回放图层（核心）**：选一个 rain episode（如 2019-06-04 Houston），显示 BG 按模型排序的分位着色；切换 forecast / observed 两种模式；叠加真实索赔 BG（空心描边）。对应 SCENARIO_REPORT 的 countdown 片段。
2. **县级排序图层**：cross-county 排序，用于 "先看哪几个县"。
3. **暴露底图**：BG 保单数/暴露（policies-only 基线），让观众能对照 "模型比暴露多了什么"。
4. **雨场背景**：GEE 自带的雨量数据（如 gridMET `IDAHO_EPSCOR/GRIDMET`）仅作背景参考；我们模型实际输入是本地 gridMET/MRMS/GEFS，不要暗示图上的雨场就是模型输入。
5. **SCS3 县-日风险**（CO/NM/AZ）：县面按日期着色，带时间滑块；第二期。
6. **指标面板**：事件分布图（现有 `docs/figures/event_eval_tx_*.png` 的交互版）：AUC、top-5% capture、事件聚类置信区间，来自 `scripts/94–106` 的输出，不在前端重算。

## 3b. 输入 → 中间输出 → 最终结果（mentor 要求，2026-11-10）

地图分三个并列图层组，选定同一个事件和县后可以逐层对照；每个变量点开即显示 "怎么来的、单位、来自哪个脚本"。数据表：`county_event_tx_<year>`（县 × 事件，列定义见 `scripts/112_gee_county_events.py` 文档字符串）。当前只覆盖 **observed-rain 模式**（gridMET 实测雨 + 4 km CNN / boosting，TX，S0，2017–2023，至少 10 个有索赔 block group 的事件，共 394 个，87,780 行县 × 事件）。forecast 模式（GEFS）另做。

**A. 输入**

| 变量（表中列名） | 含义与来历 | 来源 |
|---|---|---|
| `rain_cty_max`, `rain_cty_mean` | 事件两天内 gridMET 4 km 日最大雨量（mm）：县内各 block group 像元的最大值与均值 | gridMET，脚本 85 的特征 |
| `rain_sum_mean` | 两天总雨量（mm）在县内 block group 像元上的均值 | 同上 |
| **Exposure（拆成 5 个量，不是一个黑盒）** | | NFIP 清洗后保单表，`flood_pilot.exposure_matrices`，经 crosswalk 落到 2010 block group，再对县求和或加权 |
| `n_pol` | 事件当天仍有效的保单数（生效日 ≤ 当天 < 保单终止/取消日） | 清洗保单 |
| `sfha_share` | 其中评在特殊洪水危险区（A*/V* 区）的比例，按保单数加权 | 同上 |
| `single_share` | 其中单户住宅（占用码 1、11）的比例 | 同上 |
| `mean_cov` | 平均每单建筑保额（百万美元） | 同上 |
| `n_bg` | 当天有保单的 block group 数 | 同上 |
| `hist00` | 2000–2009 年该县 NFIP 索赔数 | 清洗索赔表（历史索赔先验，stage 19） |
| `prior3` | 事件前 3 年该县索赔数 | 同上 |

**A'. 输入：最终的聚合输入变量（模型真正读到的，`in_*` 列）**

上面的 exposure 是县的总量和保单加权占比，便于解释；CNN 表格分支实际读到的是每个 block group 的 9 个数（`scripts/89 tab_features`），这里取县内 block group 的**不加权均值**，保持模型自己的尺度（含 log 变换），点县时单独成段显示，并与上面的总量并列：

| 列 | 模型输入 |
|---|---|
| `in_log_pol` | log(有效保单数) |
| `in_sfha`, `in_single` | 洪水危险区占比、单户占比 |
| `in_cov` | 平均保额（百万美元） |
| `in_log_nbg` | log(1 + 县内 block group 数) |
| `in_hist00`, `in_prior3` | log1p 历史索赔（2000–2009、前 3 年） |
| `in_month_sin`, `in_month_cos` | 事件月份的圆周编码（同一事件各县相同，不画图，只在点县时列出） |

模型在使用前用训练年的均值和标准差做标准化，这里展示标准化前的值。

**B. 中间输出**

| 变量 | 含义 |
|---|---|
| `exp_cnn` | 县内各 block group 的 CNN 索赔概率之和（2 个种子均值） |
| `exp_boost` | 同上，boosting 模型 |
| `exp_pol` (= `n_pol`) | 只看暴露的基线 |

CNN 内部的 16 通道 embedding 是 block group 级的抽象特征，没有直观解释，**不展示**；展示的是它的输入（雨场、暴露）和输出（概率）。

**D. 与 Ground Truth 的对比（事件内）**

Ground truth = NFIP 在 [S, S+3] 的索赔（S0，去掉热带命名风暴索赔），事后数周才有，绝不是模型输入。单独一组图层：`n_claims`、`n_cbg`、`paid`（建筑赔付金额，美元）、`claims_per_1k`（每千张保单的索赔数）。

对比方式：设 K = 该事件中实际有索赔的县数，让每个模型标出得分最高的 K 个县，与实际对照，分四类：命中（绿）、漏报（橙）、误报（紫）、其余（灰）。三个模型各一张图（`class_cnn`、`class_boost`、`class_pol`）。面板里同时给出命中数，以及一张 "按得分从高到低标出县，捕获了多少比例的索赔" 曲线（CNN、boosting、只看保单数、随机）。全部 394 个事件合计，K 总和 3,770，命中：CNN 1,801（47.8%）、boosting 1,702（45.1%）、只看保单数 1,411（37.4%）。这是事件内排序的县级聚合结果，与 forecast 模式无关。

**C. 最终结果**

`pct_cnn`、`pct_boost`、`pct_pol`：该县在本事件所有县中的得分分位（1 = 最高）；对照 `n_claims`（实际索赔数）、`n_cbg`（有索赔的 block group 数），仅为回溯对照。`events_index` 给出每个事件的县级 / block-group AUC 与 top-k 捕获。

**展示口径的硬约束：**
- `exp_cnn` 的绝对值**没有意义**（训练时对无索赔行下采样，概率没有校准；脚本 94 也只在事件内重标定后使用）。界面只显示分位 / 排名，不显示 "概率 x%"。
- 县级排序与暴露排序在 forecast 模式下无差别（AUC 0.82 对 0.82），observed 模式也要与 `pct_pol` 并排显示，让观众看见模型相对 "只看保单数" 多出的部分。
- 不使用 `foundationType == 0`、`dur`（AGENTS.md 数据陷阱），本表也没有这些列。

## 4. 数据资产设计

**更新 2026-11-10（mentor 要求）：展示聚合到县级。** NFIP 的 block-group 编码混有 2010/2020 两种版本，因此展示单元改为 county（FIPS 前 5 位，跨版本稳定，不需要 crosswalk）。县分数 = 县内 block-group 分数之和（与 `scripts/94` 一致）。下表里的 BG 几何与 BG 级排序资产降级为可选（以后如要下钻再做）；§1 中 "离散 block-group 着色" 的建议在县级展示下不再适用，改为按县面着色并注明 block-group 排序仍是模型的真实输出单位。县几何由 `scripts/110_gee_county_geometry.py` 生成（TX/LA/FL/CO/NM/AZ 共 497 县，0.32 MB，小到可以用 Python API 直接上传，不需要 GCS）。

资产放在 `projects/<PROJECT>/assets/wxrisk/`。

| 资产 | 类型 | 内容 | 规模 |
|---|---|---|---|
| `bg_tx_2010` | FeatureCollection | Texas 2010 block-group 几何 + GEOID + 静态暴露（保单数等） | 约 1.5 万面，一次性 |
| `county_tx` | FeatureCollection | 县几何 + FIPS | 254 |
| `rank_{mode}_{year}` | FeatureCollection（无几何，仅属性）或带几何 | `ep`（episode id）、`geoid`、`score_pct`（事件内分位）、`rank`、`claimed`（0/1，仅回溯评估） | 每年按 episode×BG 行数分片 |
| `events_index` | FeatureCollection（无几何） | episode 元数据：日期、主要县、模式可用性、该事件 AUC / top-5% | 数百行 |
| `scs3_county_day` | FeatureCollection | 县 FIPS、日期、分位 | 第二期 |

设计要点：

- **几何与分数分离**：几何只上传一次；每次事件着色用 `ee.Join`/按 `geoid` 映射（`ee.Dictionary` + `remap`）把分数贴到几何上，避免重复上传大几何。事件数据量大时按年/事件分片，每片单资产远低于 EE 的资产大小限制。
- **BG 编码**：NFIP 的 12 位 GEOID 为混合年份，**必须经 `wxrisk.bg_crosswalk` 统一到 2010 BG**，几何也用 2010 版本（与现有 pilot 一致）。几何来源用 Census TIGER 2010 BG（本地已有时复用；否则走 `wxrisk.fetch`，不要 curl）。
- **隐私/合规**：索赔只以 BG 聚合标志展示，不展示点位（NFIP 经纬度本来也只到 1 位小数）。
- 评估标签（`claimed`）仅用于回溯展示，界面标注 "retrospective"。

## 5. 开发步骤

| 步 | 内容 | 产出 | 完成标准 |
|---|---|---|---|
| 0 | 账号与项目：建 Google Cloud 项目，在 `https://code.earthengine.google.com/register?project=<PROJECT>` 注册（选非商业或商业，决定于用途/资助），启用 Earth Engine API，建 GCS bucket；本机 `uv add earthengine-api`（放 optional extra `gee`，不进核心依赖），`earthengine authenticate`；`ee.Initialize(project=...)` 通过。**核对本方案里用到的数据集 ID（gridMET、TIGER BG）在目录里的当前名称。** | 项目 ID、bucket 名写入 `docs/GEE_PLAN.md` 的 §7 | `ee.Number(1).getInfo()` 成功 |
| 1 | 导出脚本 `scripts/110_gee_export.py`（沿用 AGENTS.md：docstring 写输入输出与命令；经 `wxrisk.fetch`/官方 CLI 而非 shell 下载；Polars/DuckDB 懒读 parquet，**注意空 key 行为**）。输出 BG 几何(GeoJSON/SHP)、`rank_*` CSV、`events_index` CSV 到 `$WXRISK_DATA/gee_export/` | 导出文件 + 行数核对 | 行数与 `flood_pilot` 表一致；GEOID 全部 12 位且在 2010 BG 内 |
| 2 | 上传脚本 `scripts/111_gee_upload.py`：`gsutil`/GCS 客户端传文件，`earthengine upload table --asset_id=...`（官方命令；CSV 用 manifest 指定属性类型，`geoid` 必须按**字符串**导入，避免前导零丢失）；等任务完成并校验 `size()` | 资产就绪 | 资产 `size()` 等于导出行数 |
| 3 | Code Editor 原型：BG 面按 `score_pct` 着色（5 档分位调色），事件选择下拉，模式切换，索赔描边，点击 BG 弹出信息（rank、暴露、claimed） | `gee/app.js`（放仓库 `gee/`，脚本源码入库，资产与数据不入库） | Houston 2019-06-04 回放与 `countdown_20190604.mp4` 的排序一致 |
| 4 | 指标面板：`ui.Chart` 或静态图 + 表，事件 AUC 分布；图例与模式说明文字 | 同上 | 数值与 `scripts/94–106` 输出逐项一致 |
| 5 | 发布为 Earth Engine App，设共享权限；资产共享权限与之匹配（App 用户需能读资产） | App 链接 | 无登录或指定人员可打开 |
| 6 | 一致性测试：`tests/test_gee_export.py`，不连网，只测导出表的 schema、GEOID 格式、分位范围、行数守恒 | pytest 通过 | `uv run python -m pytest` |
| 7（可选） | 阶段 B 自建网页 + 服务账号后端；SCS3 县-日图层 | | |

## 6. 风险与注意

- **展示的可信度**：forecast mode 仅略优于暴露排序（县级排序与暴露无差异：AUC 0.82 vs 0.82）。界面首屏必须写清；不要用 observed-rain 的 0.86 做 "预报能力" 标题。
- **配额与资产大小**：EE 对表上传、资产数量与存储有配额（以官方当前文档为准）；先估算 `rank_*` 总量，必要时只发布大事件（≥10 个有索赔 BG 的事件，与 RESULTS 评估口径一致）。
- **GEE 只在 Python 端用于上传，不在训练代码里引入**：保持 `torch`/`ee` 都是 optional extra，核心环境不变。
- **凭据**：不入库；服务账号 JSON 与 `earthengine` 令牌放 `$HOME`，`.gitignore` 已排除数据，凭据另需确认不在被追踪目录。
- **时间滑块**：EE 无原生时间序列属性，事件切换用下拉/滑块调用 `filter(ep == ...)`，每次回放是一次服务器端过滤，不是预载动画。
- 新增数据源与脚本后，按 AGENTS.md 更新 `docs/DATA_SOURCES.md`，并在 `RESULTS.md` 记一个 "展示层" 小节（非实验，但写清 Caveats）。

## 7. 待确认

| 项 | 默认假设 | 需要谁决定 |
|---|---|---|
| Google Cloud 项目 ID / bucket | 新建专用项目 | 用户 |
| EE 注册类型（非商业 vs 商业） | **已定（2026-11-10）：学校账户已注册非商业** | 已确认 |
| 受众与公开程度 | **已定：内部展示优先**，走阶段 A（Earth Engine App，按指定人员/学校域共享）；阶段 B 暂缓 | 已确认 |
| 学校账户下的项目 ID | `impact-foreacsting-flooding`（用户提供 2026-11-10，拼写含 "foreacsting"，待 `ee.Initialize` 验证是否为真实 ID） | 用户 |
| GCS bucket | 待确认：学校账户若无权创建，则改用 Python API 直接上传的替代路径 | 用户 |
| 展示范围 | 先 Texas 洪水线，SCS3 第二期 | 用户 |
