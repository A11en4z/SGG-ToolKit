# BC-RPCM：在 RPCM 上用双向条件 Transformer 替换特征融合（不讨论原型学习）

## 1. 目标与边界

### 1.1 目标

在现有 RPCM（STAR 论文的融合阶段）基础上：

- 保留 RPCM 的“通路优势”（语义-视觉对齐、门控融合、union 修正、以及基于图结构的特征更新）。
- 用更先进的 Transformer 结构替换 **RPCM 的特征融合部分**，以更好地建模 `subject/object/union` 之间的高阶交互。
- 预留一个 **外部谓词知识特征** `k` 的注入接口（暂不讨论知识来源与训练方式，只假设该特征可用）。
- 明确把“特征融合”与“原型学习/原型损失”分割为两个阶段，便于后续单独改原型学习。

### 1.2 边界（本稿不做的事）

- 不修改 RPCM 的原型学习算法（prototype construction / cosine logits / 各类 prototype regularization loss）。
- 不改变数据、采样、推理后处理等其它模块逻辑。
- 不承诺改动后一定更好；本稿提供的是实现级设计与可做的消融。

## 2. 现有代码定位与当前 RPCM 的融合路径

### 2.1 关键代码入口

RPCM predictor 在下面文件中实现：

- `/gz-data/SGG-ToolKit/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
  - `@registry.ROI_RELATION_PREDICTOR.register("RPCM")`
  - `RPCM.forward(...)`

Transformer 系列 predictor 对比用：

- 同文件中的 `TransformerPredictor.forward(...)`
- `model_transformer.py` 的 `TransformerContext.forward(...)`
- `model_Hybrid_Attention.py` 的 `SHA_Context.forward(...)`

### 2.2 RPCM.forward 的阶段划分（建议的“分割点”）

RPCM.forward 的整体流程（以 predcls/sgcls/sgdet 的共同部分描述）：

1) **Pairwise 特征提取 + 图结构消息传递（GCN）**  
输入：`roi_features`（实例 RoI 特征）、`union_features`（pair 的 union 特征）、`proposals`、`rel_pair_idxs`  
输出：更新后的 `roi_features`、`union_features`  
代码段：`pairwise_feature_extractor(...)`、`_get_map_idxs(...)`、`gcn_collect_feat/gcn_update_feat`。

2) **refine object labels（对象预测/重分类）**  
输出：`entity_dists`（obj logits）、`entity_preds`（obj label 预测）、`obj_labels`（gt 或 None）

3) **特征融合（本文要替换的部分）**  
从更新后的 `roi_features` 和 `union_features` 以及 GloVe 语义 embedding，构造关系表示 `rel_rep1`。  
核心包括：
   - 语义投影：`Ws/Wo` 作用于 `entity_embed`（词向量）
   - 视觉投影：`vis2sem(h)` 作用于 `sub_rep/obj_rep`
   - 门控融合：`gate_sub/gate_obj`
   - 二元融合算子：`fusion_func(sub, obj)`
   - union 修正：`rel_rep1 = fusion_so - gate * h(union)`

4) **原型学习与关系分类（本文不讨论）**  
把 `rel_rep1` 投到 `project_head` 空间后，与 predicate prototype 做 cosine 相似度得到 `rel_dists`，并在训练时加入 prototype 相关损失。

建议的“分割点”：以 `rel_rep1` 为边界。

- 上半部分：`compute_fusion_features(...) -> rel_rep1`（本文重点）  
- 下半部分：`prototype_head(rel_rep1, rel_labels, ...) -> rel_dists, add_losses`（保持现状，后续再改）

### 2.3 现有融合公式（便于你对齐“通路优势”）

在现有 RPCM 中（伪公式对应源码变量）：

**(A) 语义-视觉对齐 + 门控（对 subject/object）**

- `s_embed = W_sub( t_s )`，`o_embed = W_obj( t_o )`  
- `sem_sub = vis2sem( x_s )`，`sem_obj = vis2sem( x_o )`  
- `gate_sem_sub = sigmoid( gate_sub([s_embed, sem_sub]) )`  
- `gate_sem_obj = sigmoid( gate_obj([o_embed, sem_obj]) )`  
- `sub = s_embed + gate_sem_sub ⊙ sem_sub`  
- `obj = o_embed + gate_sem_obj ⊙ sem_obj`

**(B) 二元融合算子**

- `fusion_so = fusion_func(sub, obj)`（源码为 `relu(x+y) - (x-y)^2`）

**(C) union 修正（抑制 union 中的“冗余视觉共现”）**

- `sem_pred = vis2sem( down_samp(union_features) )`
- `gate_sem_pred = sigmoid( gate_pred([fusion_so, sem_pred]) )`
- `rel_rep1 = fusion_so - gate_sem_pred ⊙ sem_pred`

本文的 BC-RPCM 设计会把上面 (A)(B)(C) 的归纳偏置保留为：

- token 初始化时保留 (A) 的“语义-视觉对齐 + 门控”；
- 在 Transformer block 内显式引入 `union token` 参与交互；
- 输出关系 token 时保留 (C) 的“union 修正/反证据”倾向（可实现为 gated subtraction 或者 learnable anti-correlation head）。

## 3. 现有 TransformerPredictor 融合为何弱于 RPCM（结合本代码库实现）

对比本仓库当前实现：

### 3.1 TransformerPredictor 的“融合通路”结构

1) `TransformerContext/SHA_Context` 输出的是 **对象级 edge_ctx**（本质是对每个对象的上下文编码结果），没有显式的 relation pair token。
2) 在 predictor 里把 `head_rep/tail_rep` 取出后，对于每个 pair 仅做：

- `prod_rep = concat(head_rep[sub_idx], tail_rep[obj_idx])`
- `ctx_gate = post_cat(prod_rep)`
- `visual_rep = ctx_gate * union_features`
- `rel_logits = rel_compress(visual_rep) + ctx_compress(prod_rep)`

这条通路的特点是：

- union 特征只被 **乘法门控**，缺少 RPCM 的“union 修正（减法反证据）”；
- pair 交互只在 `concat + MLP` 层面发生，缺少显式的 `S/O/U` 多方交互建模；
- 对语义（词向量）信息的利用主要在 context layer 的 obj_embed，未形成 RPCM 那种“语义→视觉对齐空间”的稳定通路。

### 3.2 RPCM 的“强归纳偏置”

RPCM 的强点来自：

- **语义对齐空间**：`vis2sem` 把视觉映射到与语义原型同一尺度；
- **门控融合**：根据语义与视觉一致性调节视觉注入；
- **union 反证据修正**：`F(s,o) - gate * h(union)`，可以抑制“仅 union 共现”的伪关系；
- **图结构更新**：对象与谓词特征在融合前经过显式的 GCN message passing。

因此，想用 Transformer 替换融合架构时，关键不是“把一堆特征堆进 transformer”，而是要：

- 把 `S/O/U` 作为显式 token 建模；
- 用注意力方向/mask 保留“通路”与“反证据”机制；
- 保留语义-视觉对齐空间作为 token 初始化与输出头的一部分。

## 4. 推荐修改：BC-RPCM（Bidirectional Conditioning RPCM）

### 4.1 总体结构（最小侵入）

保持 RPCM 的以下部分不变：

- `PairwiseFeatureExtractor + GCN`（更新 roi_features/union_features）
- `refine_obj_labels`
- （暂不改）后续原型学习与 loss

仅替换“特征融合”阶段：

- 旧：`W_sub/W_obj + vis2sem + gate + fusion_func + union subtraction`
- 新：`BCB（Bidirectional Conditioning Block）`，输入 `S/O/U`（可选 `K`），输出 `rel_rep1`（以及可选的更新后 `sub/obj` 表示）

### 4.2 Token 设计（保持 RPCM 语义-视觉对齐）

对每个关系对 `(s,o)` 构造 token（维度统一为 `D = mlp_dim`，即 RPCM 中的 `2048`）：

- `tS`（subject token）：来自 `s_embed` 与 `sem_sub` 的门控融合
- `tO`（object token）：来自 `o_embed` 与 `sem_obj` 的门控融合
- `tU`（union token）：来自 `sem_pred`（即 `vis2sem(down_samp(union_features))`）
- `tK`（knowledge token，可选）：来自外部谓词知识特征 `k` 的线性投影到维度 `D`

初始化建议（复用现有层，保证训练稳定）：

- `tS0 = norm( residual( sub ) )`（保留现有 `linear_sub + relu + dropout + layernorm` 风格）
- `tO0 = norm( residual( obj ) )`
- `tU0 = norm( residual( sem_pred ) )`（可新增 `linear_union/norm_union`，或复用 `linear_pred/norm` 结构）
- `tK0 = proj_k(k)`，再做 layernorm

### 4.3 BCB：双向条件 Transformer block（核心）

BCB 的核心想法：让 `S/O` 与 `U/K` **双向调制**，并显式地产生关系 token `R`。

推荐一个实现友好的最小版本（单层或少量层数）：

**(1) Token 堆叠**

对每个 pair 构造序列：

- 无知识：`[tS, tO, tU]`
- 有知识：`[tS, tO, tU, tK]`

序列长度很短（3 或 4），因此 transformer 计算量很小，适合在 relation head 内使用。

**(2) 注意力方向约束（保留“通路优势”）**

为了避免退化为“无归纳偏置的混合”，建议引入 attention mask（或用结构化 cross-attn）实现以下信息流：

- `S/O` 允许 attend 到 `U/K`（union/knowledge 调制实体表征）
- `U` 允许 attend 到 `S/O`（实体上下文反向调制 union 表征）
- `S` 与 `O` 的直接互注意力可以：
  - 方案 A：允许（更灵活，但可能更易过拟合）
  - 方案 B：禁止（把 S↔O 交互更多交给 U/R 介质）

这就是 “Bidirectional Conditioning” 的最小落地：实体与 union/知识互为条件。

**(3) 输出关系 token（替代 fusion_func + union subtraction）**

输出方式推荐两种（都能保留 RPCM 的减法反证据倾向）：

- 方案 1（显式反证据）：  
  - 先得到更新后的 `tS1,tO1,tU1(,tK1)`  
  - `fusion_so = g( [tS1,tO1] )`（用小 MLP / bilinear）  
  - `rel_rep1 = fusion_so - sigmoid(h([fusion_so,tU1])) ⊙ tU1`

- 方案 2（关系专用 token）：  
  - 增加一个 learnable `tR`（relation token，类似 [CLS]）  
  - 序列变为 `[tR, tS, tO, tU(,tK)]`  
  - 让 `tR` attend 到其它 token，最终 `rel_rep1 = tR_out`  
  - 再附加一个小的 `anti-union` head：`rel_rep1 = rel_rep1 - gate ⊙ tU_out`（可选）

从“贴近现有 RPCM”角度，方案 1 更接近原结构，迁移更稳；方案 2 更像标准 transformer，扩展性更好。

### 4.4 外部知识接口（tK 的输入规范）

这里仅定义接口，不定义知识生成方式。

假设每个关系对都有一个外部谓词知识特征 `k_pair`：

- 形状：`(num_rel, Kdim)`  
- 语义：可以是 LLM embedding、知识图谱 embedding、规则特征等

注入位置：

- 最推荐：作为 BCB 的 `tK` token（pair 级）  
  - `tK0 = Linear(Kdim -> D)(k_pair)`  
  - 再 `LayerNorm`

若你未来希望“知识与谓词类别”强绑定（类别级知识，而非 pair 级）：

- 可以把 `k` 设计为 predicate class 的 bank：`K_bank (num_rel_cls, Kdim)`，再根据当前 pair 的候选谓词分布做加权，得到 `k_pair`。
- 这属于“知识与原型学习结合”的方向，本文先不展开。

### 4.5 与原型学习的分割方式（实现层面建议）

为了后续更方便改原型学习，建议把 RPCM.forward 拆成两个纯函数式子流程：

1) `forward_fusion(...) -> (entity_dists, entity_preds, rel_rep1, num_objs, num_rels, pair_pred)`  
2) `forward_prototype(rel_rep1, rel_labels, pair_pred, ...) -> (rel_dists, add_losses)`

其中 `forward_prototype` 内部保持现状（包含 project_head / predicate_proto / loss_dis / l21 / dist_loss 等）。

这样你后续要把 prototype learning 换成“外部知识驱动的 predicate head”时，只需要替换第 2 步。

## 5. 实现清单（后续真正动手时按这个改）

### 5.1 主要改动文件

- `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`
  - 在 `RPCM` 内新增/替换融合模块
  - 新增 `BCB/BC-RPCM` 相关 nn.Module（建议放在 RPCM 下方，和 `MLP/fusion_func` 同区域）
- （可选）`maskrcnn_benchmark/config/defaults.py` 或相应 cfg 定义文件  
  - 增加 `MODEL.ROI_RELATION_HEAD.RPCM_FUSION_TYPE` 等开关
  - 增加 `MODEL.ROI_RELATION_HEAD.BC_RPCM.*` 超参（层数、head 数、dropout、是否启用 tK 等）

### 5.2 必需的 cfg 开关（建议）

- `MODEL.ROI_RELATION_HEAD.PREDICTOR = "RPCM"`（保持不变）
- `MODEL.ROI_RELATION_HEAD.RPCM_FUSION_TYPE`：
  - `"legacy"`：保留现有 fusion_func + union subtraction
  - `"bc_rpcm_v1"`：启用 BCB（方案 1）
  - `"bc_rpcm_v2"`：启用 relation token（方案 2）
- `MODEL.ROI_RELATION_HEAD.BC_RPCM.USE_KNOWLEDGE`：是否启用 `tK`
- `MODEL.ROI_RELATION_HEAD.BC_RPCM.K_DIM`：外部知识维度

### 5.3 需要对齐的 I/O（保持训练脚本不改）

RPCM.forward 必须最终返回：

- `entity_dists`：按 `num_objs` split 的 list[Tensor]
- `rel_dists`：按 `num_rels` split 的 list[Tensor]
- `add_losses`：dict

因此即使你只改融合模块，也要保证输出 `rel_rep1` 的形状与后续 prototype head 兼容（通常是 `(sum(num_rels), D)`）。

## 6. 建议消融与排错顺序

为了降低改动风险，建议按以下顺序实现与验证：

1) 先做“分割点重构”：把现有融合逻辑抽成 `forward_fusion_legacy`，保证结果完全一致。
2) 实现 `BCB` 但先不开 attention mask（让其退化为小 transformer），对齐训练是否收敛。
3) 加回 attention mask，观察性能与稳定性。
4) 最后再接入 `tK`，做 `USE_KNOWLEDGE` 的开关消融。

常见排错点：

- `num_objs/num_rels` split 是否与 cat 对齐；
- `union_features` 在 GCN 后的维度是否仍与 `down_samp` 匹配；
- predcls/sgcls/sgdet 模式下 `entity_preds` 的来源是否一致；
- 新模块的 LayerNorm/Dropout 是否导致 logits scale 漂移（可参考 RPCM 里已有的 `logit_scale` 做稳定化）。

