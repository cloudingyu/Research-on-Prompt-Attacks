### 《大语言模型提示词攻击防御方案》

**目录**

**1. 大预言模型提示词攻击防御方案分类**
*  1.1 提示词微调与前缀防御
*  1.2 输入纯化与重构
*  1.3 检测与架构隔离

**2.防御方案对比分析与总结**

**3.参考文献**

---

### 1.大预言模型提示词攻击防御方案分类

随着大语言模型（LLM）的广泛应用，针对其安全对齐机制的攻击手段（如Jailbreak、Prompt Injection）层出不穷。攻击者通过构造对抗性后缀、利用角色扮演或插入触发词来诱导模型输出有害内容。本文将

#### 1.1 提示词微调与前缀防御 (Prompt Tuning & Prefix Defense)

**来源文献：** Fight Back Against Jailbreaking Via Prompt Adversarial Tuning [1]

该类方法的核心思想是不改变模型参数，而是通过优化一个“防御性前缀”或“系统提示词”，使其能够抵消攻击性指令的影响。
	![[image/Pasted image 20251210154105.png]]
- **图解描述：** 图中展示了推理阶段的流水线。
    - Unprotected LLM: 用户输入攻击Prompt -> 模型输出炸弹制作教程。
    - Protected LLM via PAT: 系统自动将 **{Defense Control}** 拼接到用户Prompt前 -> 模型输出 "I'm sorry..."。

**方案：Prompt Adversarial Tuning (PAT)**

* **原理**：  
	PAT 的核心在于**双层优化（Bi-level Optimization）**。它不微调整个模型，而是训练一个“防御前缀”（Defense Control）。训练过程模拟了一场博弈：
	1. **攻击者视角（更新攻击后缀）：** 固定防御前缀，寻找能让模型**突破防御**、输出有害内容的攻击后缀（类似 GCG 攻击算法）。
	2. **防御者视角（更新防御前缀）：** 固定刚才生成的强力攻击后缀，寻找能让模型**拒绝回答**（输出 "I am sorry"）且能正常回答良性问题的防御前缀。
    ![[image/Pasted image 20251210153613.png]]
	通过这种反复迭代，防御前缀“见过”了各种强大的攻击形式，从而获得了极强的鲁棒性。
	
	**目标：** 使模型在面对恶意攻击时输出拒绝响应（如 "I am sorry..."），而在面对正常查询时保持原有响应。
	
* **实现逻辑（伪代码）**：
	基于Fight Back Against Jailbreaking Via Prompt Adversarial Tuning[1]中的Algorithm 2 (Prompt Adversarial Tuning)：
```python
# 输入: 恶意Prompt集合, 良性Prompt集合, 初始攻击前缀, 初始防御前缀
# 输出: 优化后的防御前缀

def PAT_Training(harmful_data, benign_data, defense_control, attack_control, iterations):
    for t in range(iterations):
        # 1. 更新攻击控制 (攻击者视角)
        # 固定防御前缀，寻找能让模型输出有害内容的攻击后缀
        grads_attack = compute_gradients(loss_function(target="harmful"))
        candidates_attack = generate_candidates(grads_attack)
        attack_control = select_best_candidate(candidates_attack)

        # 2. 更新防御控制 (防御者视角)
        # 固定攻击后缀，寻找能让模型输出拒绝内容且保持良性问答的防御前缀
        grads_defense = compute_gradients(loss_function(target="refusal"))
        candidates_defense = generate_candidates(grads_defense)
        
        # 选择最佳防御前缀：既能拒绝恶意攻击，又能回答良性问题
        defense_control = select_best_candidate_weighted(
            candidates_defense, 
            weight_benign=alpha, 
            weight_defense=(1-alpha)
        )
    
    return defense_control
```

* **抵御攻击的可能性分析：**
	- **防御效果：** 极高。实验显示，在Vicuna-7B和Llama-2上，PAT将高级攻击（如GCG, AutoDAN）的攻击成功率（ASR）降至接近 **0%**。
		![[image/Pasted image 20251210154248.png]]
	- **优势：** 部署成本极低（只需加前缀），不影响模型推理效率。
	- **局限：** 对抗性训练过程较慢，且如果攻击者知道防御前缀的具体内容，可能会进行针对性的自适应攻击（Adaptive Attack）。

 ---

#### 1.2 输入纯化与重构 (Input Purification)

**来源文献：** TAPDA: Text Adversarial Purification as Defense Against Adversarial Prompt Attack for Large Language Models[2]

该类方法假设输入中包含对抗性噪声，试图通过特定的机制“清洗”输入，去除恶意触发词，恢复原始语义。

**方案：Text Adversarial Purification (TAPDA)**

* **原理：**  
    对抗性 Prompt（尤其是像 GCG 这种生成的乱码后缀）是非常**脆弱**的。它们依赖于特定的 Token 组合来触发模型的 Bug。
	- **机制：** TAPDA 利用 LLM 的能力，对输入 Prompt 进行**随机掩码（Masking）**，然后让模型填空（Predict），试图恢复原句。
	- **效果：** 对于正常的语义句子，填空能恢复原意；但对于对抗性后缀，随机掩码和重构会**破坏其精心设计的攻击结构**。最后通过投票（Voting）和困惑度（PPL）筛选出最通顺（PPL最低）的 Prompt，这通常就是去除了攻击性的 Prompt。
	![[image/Pasted image 20251210161145.png]]
- **图解描述：**
    - 输入：Malicious Instruction + Adversarial Suffix (红色块)。
    - 处理：生成n个 Masked prompts (例如把后缀里的词 Mask 掉)。
    - 预测：LLM 填空。
    - 筛选：计算 `SvotingSvoting​`(投票分) 和`SPPLSPPL​`(困惑度分)。
    - 输出：Purified Prompt (此时红色的对抗后缀已经被破坏或替换成了无害内容)。

	![[image/Pasted image 20251210162145.png]]
*  **实现逻辑 (伪代码)：**
	基于TAPDA: Text Adversarial Purification as Defense Against Adversarial Prompt Attack for Large Language Models中的Algorithm 1:
```python
def TAPDA(prompt P, num_samples n, mask_ratio):
    masked_prompts = []
    # 1. 生成 n 个被随机掩码的 Prompt
    for i in range(n):
        masked_prompts.append(random_mask(P, mask_ratio))
    
    predictions = []
    # 2. 利用 LLM 预测被掩码的位置
    for p_masked in masked_prompts:
        predictions.append(LLM_predict(p_masked))
        
    # 3. 投票机制确定候选 Token
    candidate_prompt = majority_voting(predictions)
    
    # 4. 基于困惑度 (PPL) 的筛选与加权
    # 结合投票得分和PPL得分选择最佳的纯化后 Prompt
    final_prompt = select_best_by_score(candidate_prompt, weight_lambda)
    
    return final_prompt
```

- **抵御攻击的可能性分析：**
    - **防御效果：** 中等到高。对于 AdvPrompter 攻击，ASR 从 72.6% 降至 **33.9%**；对于 AutoDAN 和 GCG 也有显著效果。
	    ![[image/Pasted image 20251210162311.png]]
    - **优势：** 保持了 Prompt 的可读性，不需要额外的外部模型，利用 LLM 自身能力。
    - **局限：** 推理延迟较高（需要多次预测和聚合），对于语义极其敏感的 Prompt 可能会改变原意。

---

#### 1.3 检测与认证机制 (Detection & Certification)

该类方法侧重于在输入进入模型核心处理流程前，识别并拦截恶意输入。

**方案A：UniGuardian (基于损失值波动的检测)**

* **来源文献：** UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models.[3]

- **原理：**  
	UniGuardian 利用了对抗攻击的一个弱点：**敏感性**。
    UniGuardian 定义了“提示触发攻击”（Prompt Trigger Attacks, PTA）。它发现，如果输入中包含触发攻击的关键词（Trigger），移除这些词会导致模型生成的 Loss 发生剧烈变化（高 Z-score）。  
    ![[image/Pasted image 20251210164351.png]]
    **方法：** 随机掩盖输入中的单词，观察 Loss 的变化。如果掩盖某些词导致 Loss 异常波动，则判定为恶意输入。采用 "Single-Forward Strategy" 提高效率。
    ![[image/Pasted image 20251210164650.png]]
	 **图解描述：**
        -  图 2(b) 展示了 **Trigger Detection**  过程：通过 "Randomly Generate Index Tuples" 生成掩码，计算 `Z-Scores of Si​`。
        - 图 2(c) 展示了 **Single-Forward Strategy**：为了快，它把原始 Prompt 和所有掩码后的 Prompt 拼成一个大 Batch，做一次前向传播就算出结果，不用跑很多次。
    
- **实现逻辑 (描述)：**
    1. **基准生成：** 让模型基于原始 Prompt 生成一段文本，计算其 Logits。
    2. **构建掩码样本：** 复制原始 Prompt `n`次，每次随机掩盖不同的单词子集。
    3. **单次前向传播：** 将所有掩码样本拼接，一次性输入模型。
    4. **计算不确定性分数：** 对比掩码样本与基准生成的 Logits 差异（计算 Z-score）。
    5. **判定：** 如果某次掩码导致的不确定性分数超过阈值（Suspicion Score 高），则标记为攻击。

- **抵御攻击的可能性分析：**
    - **防御效果：** 极高。在 Prompt Injection 和 Backdoor 攻击上表现都很好，尤其在面对Backdoor攻击时，AUROC 接近 **0.99**。
	    ![[image/Pasted image 20251210185405.png]]
    - **优势：** 无需训练（Training-free），通用性强（同时防御后门、对抗攻击和提示注入）。
    - **局限：** 需要访问模型的 Logits（白盒或灰盒），纯黑盒 API 可能无法使用。

**方案B：Erase-and-Check (擦除与检查认证)** 

* **来源文献：** Certifying LLM Safety Against Adversarial Prompting[4]

- **原理：**  
    这是一个提供**数学认证（Certified Guarantee）** 的方法。  
	攻击者通常是在一个有害指令（比如“制造炸弹”）后面加一段乱码后缀。
	- **机制：** 防御者不需要知道哪部分是后缀。防御者只需要**逐个长度擦除**输入的末尾。
	- **逻辑：** 假设攻击后缀长度不超过 `d`。从删去1个字符开始，一直试到删去 `d`个字符。在这个过程中，必然有一次，会**恰好把攻击后缀删干净**，只剩下原始的有害指令（“制造炸弹”）。此时，安全过滤器（Safety Filter）一定能识别出这是有害的，从而拦截请求。
    ![[image/Pasted image 20251210190716.png]]
	- **图解描述：**  
    图中展示了一个具体的例子：
    - Input: "Harmful Prompt" + "Adversarial Tokens" (蓝色+红色块)
    - Process: **Erase**（逐行擦除末尾的红色块） -> **Check** (送入 Safety Filter) -> **Result** (只要有一行由 Safety Filter 报红，最终结果就是 Harmful)。

	![[image/Pasted image 20251210190821.png]]
* **实现逻辑 (伪代码)：**  
	
	```python
	def Erase_and_Check(prompt P, max_erase_length d, safety_filter):
    # 1. 检查原始 Prompt
    if safety_filter.is_harmful(P):
        return True # 有害
        
    # 2. 逐个擦除后缀或子序列进行检查
    for i in range(1, d + 1):
        # 生成擦除后的子序列 (例如擦除末尾 i 个 token)
        subsequence = P[:-i] 
        
        if safety_filter.is_harmful(subsequence):
            return True # 只要发现一个有害子序列，即判定为攻击
    
    return False # 安全
	```

- **抵御攻击的可能性分析：**
    - **防御效果：** 提供理论认证。如果攻击后缀长度小于`d`，且原始有害 Prompt 能被识别，则该防御**保证**能拦截攻击。原文中提到实验中对 AdvBench 的认证准确率达到 **92%-100%**。
    - **优势：** 具有数学上的安全证明，不仅仅是经验有效。
    - **局限：** 计算开销大（需要检查很多子序列），随着擦除长度增加，可能会误伤良性 Prompt（降低 Utility）。

---

#### 1.4 架构隔离防御 (Architectural Defense)

**来源文献：** Securing with Dual-LLM Architecture: ChatTEDU an Open Access Chatbot’s Defense[5]

该类方法通过系统架构设计，引入额外的 LLM Agent 来专门负责安全审核。

**方案：Dual-LLM Architecture (双模型架构)**

- **原理：** 
    生成式 LLM（如 GPT-4）为了回答问题，倾向于顺从用户指令，这导致容易被绕过。该方案采用基于 Agent 的工作流，将系统分为两个独立的 LLM：
    - **LLM-1 (Input Guard):** 专门负责意图识别和安全过滤。只有被判定为安全的 Query 才会传递给下一步。
    - **LLM-2 (Response Generator):** 负责基于知识库生成回答。配合 mTLS 和 JWT 等网络层安全措施。

- **实现逻辑 (架构描述)：**
	![[image/Pasted image 20251210192008.png]]
    1. **路由层：** 接收用户 Query。
    2. **LLM-1 分析：** 使用专门的 Prompt（包含 Few-shot 攻击示例）判断 Query 是否包含 Prompt Injection、Jailbreak 或垃圾信息。
    3. **分支处理：**
        - 恶意： 记录日志，直接拒绝或返回预设消息。
        - 良性： 传递给 LLM-2。  
    4. **LLM-2 生成：** 结合 RAG（检索增强生成）生成教育场景下的回复。  
    5. **响应处理：** 再次过滤输出并返回给用户。

- **抵御攻击的可能性分析：**
    - **防御效果：** 在真实世界的大学聊天机器人部署中，拦截了 **100%** 的已知攻击（180次尝试），误报率仅 0.28%。
	    ![[image/Pasted image 20251210192356.png]]
    - **优势：** 架构清晰，易于工程落地，隔离了风险（LLM-2 不会接触恶意 Prompt）。
    - **局限：** 增加了系统延迟（Latency 增加了约 18%）和 API 调用成本（双倍 Token 消耗）。

---

### 2.对比分析与总结

#### 2.1 不同防御方案对比分析

| 防御方案                                | 核心机制                                 | 抵御攻击能力分析                                                            | 适用场景               |
| ----------------------------------- | ------------------------------------ | ------------------------------------------------------------------- | ------------------ |
| **PAT** (Prompt Adversarial Tuning) | **前缀微调**：通过对抗训练生成防御前缀                | **极高**。针对特定攻击微调，能有效防御 GCG/AutoDAN 等高级攻击，且保留较高良性问答率。                 | 模型开发者，具备白盒/微调权限    |
| **UniGuardian**                     | **异常检测**：基于随机掩码造成的 Loss 波动 (Z-score) | **极高**。不仅防御 Jailbreak，还能检测 Prompt Injection 和 Backdoor。AUROC ~0.99。 | 模型部署阶段，需访问 Logits  |
| **Erase-and-Check**                 | **可信认证**：擦除 Token 子序列并检查             | **高 (可认证)**。提供数学上的安全保证。只要攻击长度在阈值内，理论上 100% 拦截。                      | 对安全性要求极高的场景        |
| **Dual-LLM** (ChatTEDU)             | **架构隔离**：独立的审核 LLM + 生成 LLM          | **高 (实战验证)**。在真实部署中实现了 100% 拦截。利用 Agent 隔离风险。                       | 企业级应用，对延迟不极度敏感     |
| **TAPDA**                           | **输入纯化**：掩码预测 + 投票重构                 | **中高**。有效破坏对抗性后缀的结构，尤其是针对 AdvPrompter 这类语义通顺的攻击。                    | 通用 API 调用，无需模型内部参数 |

#### 2.2 综合建议

- 对于**模型开发者**，建议在推理阶段集成 **PAT** 的防御前缀。
- 对于**应用开发者**（调用 API），**Dual-LLM** 架构是最稳健的工程化落地通过方案；如果考虑到 Token 成本，可以采用 **Erase-and-Check** 的变体（如 GreedyEC）作为轻量级过滤器。
- 对于**模型私有部署权限**，**UniGuardian** 提供了最先进的无训练检测能力。

#### 2.3 综合防御架构设计

* **2.3.1 架构设计理念**
	本方案将防御流程划分为三个阶段：**输入检测层（Input Detection）**、**输入净化层（Input Purification）** 和 **核心推理层（Core Inference）**。
	- **轻量级优先：** 优先使用计算开销小的方法（如 UniGuardian 的单次前向）进行过滤。
	- **按需净化：** 只有当检测到疑似攻击但又不确定时，才启动开销较大的净化（TAPDA）或认证（Erase-and-Check）流程。
	- **隔离执行：** 核心模型始终处于“被保护”状态，不直接接触原始用户输入。

* **2.3.2 架构流程图：** 
```
graph TD
    UserInput[用户原始输入] --> Layer1
    
    subgraph Layer1 [第1层：异常检测与初筛]
        UniGuardian[UniGuardian 异常检测] -- Loss Z-score 高 --> Block[直接拒绝]
        UniGuardian -- Loss Z-score 中 --> Layer2
        UniGuardian -- Loss Z-score 低 --> Layer3
    end

    subgraph Layer2 [第2层：深度清洗与认证 (针对可疑输入)]
        TAPDA[TAPDA 文本纯化] --> Reconstruct[重构输入]
        Reconstruct --> EraseCheck[Erase-and-Check 认证]
        EraseCheck -- 有害 --> Block
        EraseCheck -- 安全 --> Layer3
    end

    subgraph Layer3 [第3层：安全推理执行]
        Prefix[加载 PAT 防御前缀] --> GuardLLM[LLM-1: 意图审核 Agent]
        GuardLLM -- 判定恶意 --> Block
        GuardLLM -- 判定安全 --> CoreLLM[LLM-2: 核心生成模型]
    end

    CoreLLM --> FinalOutput[最终响应]
```

* **2.3.3 详细模块实现**

	**模块 A：基于 UniGuardian 的“看门人” (The Gatekeeper)**
	- **来源：** UniGuardian
	- **作用：** 作为第一道防线，快速识别是否存在触发词（Trigger）。
	- **策略：** 利用 UniGuardian 的 "Single-Forward Strategy"，对输入进行一次并行的掩码推理。如果发现移除某些词导致 Loss 剧烈波动，说明存在 Prompt Injection 或 Jailbreak 攻击特征。
	- **配置：** 设置一个较高的敏感度阈值，宁可误判为“可疑”（进入Layer 2），也不漏放。

	**模块 B：基于 TAPDA + Erase-and-Check 的“清洗站” (The Sanitizer)**
	- **来源：** TAPDA & Erase-and-Check
	- **作用：** 处理那些通过了第一层但仍被标记为“可疑”的输入。
	- **组合逻辑：**
		1. 先运行 **TAPDA**：利用 LLM 的 mask-predict 能力破坏可能存在的对抗性后缀结构，重构出语义通顺的 Prompt。
	    2. 再运行 **Erase-and-Check**：对纯化后的 Prompt 进行最终安全认证。因为 TAPDA 已经去除了大部分噪声，此时 Erase-and-Check 可以设置较小的擦除长度 `d`（如 d=5），从而在保证安全的同时减少计算量。

	**模块 C：基于 PAT + Dual-LLM 的“安全核心” (The Secure Core)** 
	- **来源：** PAT &Dual-LLM
	- **作用：** 最终的业务处理与生成。
	- **组合逻辑：**
	    1. **架构隔离 ：** 系统部署两个 LLM。LLM-1 负责最终的语义审核（比如识别 UniGuardian 没看出来的社会工程学攻击）；LLM-2 负责回答。
	    2. **前缀加固 ：** 在 LLM-2（核心回答模型）的 System Prompt 中，强制加入由 **PAT** 算法训练出来的 Defense Control 前缀。即使前两层防御都被绕过，这个经过对抗训练的前缀也能极大增加模型拒绝回答恶意问题的概率。


* **2.3.4 方案优势分析**

	这个综合方案解决了单个方案存在的“短板”：

	1. **解决“误伤”与“漏判”的矛盾：**
	    - 单纯使用 Erase-and-Check虽然安全但太慢；单纯使用 PAT (Paper 1) 虽然快但可能被新型后缀绕过。
	    - 该方案利用 UniGuardian进行快速分流，只对可疑样本进行重计算，极大提高了平均吞吐量（Throughput）。
        
	2. **解决“对抗性后缀”与“语义攻击”的双重威胁：**
	    - TAPDA (Paper 5) 和 UniGuardian (Paper 2) 非常擅长处理乱码式的对抗后缀（如 GCG 攻击）。
	    - Dual-LLM (Paper 4) 的 Agent 机制擅长理解语义层面的攻击（如角色扮演、PUA模型）。
	    - 两者结合实现了全覆盖。
        
	3. **提供“兜底”机制：**
	    - 即使攻击者绕过了检测和清洗，最终送入核心模型的 Prompt 依然带着 PAT (Paper 1) 的防御前缀。这个前缀是通过梯度优化生成的，能从模型内部激活拒绝机制，作为最后的保险锁。


---

### 参考文献

[1]Mo, Yichuan et al. "Fight Back Against Jailbreaking Via Prompt Adversarial Tuning",Conference on Neural Information Processing Systems (2024)[链接](https://arxiv.org/abs/2402.06255)

[2]H. Yang, "TAPDA: Text Adversarial Purification as Defense Against Adversarial Prompt Attack for Large Language Models," 2025 8th International Conference on Advanced Algorithms and Control Engineering (ICAACE), Shanghai, China, 2025, pp. 1998-2004, doi: 10.1109/ICAACE65325.2025.11020575. keywords: {Computer vision;Filters;Control engineering;Purification;Large language models;Computational modeling;Aggregates;Safety;Text Adversarial Purification;Adversarial Attack;Trustworthy AI},
[链接](https://ieeexplore.ieee.org/document/11020575)

[3]Lin, Huawei et al. "UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models.",Computing Research Repository abs/2502.13141 (2025)[链接](https://arxiv.org/abs/2502.13141)

[4]Kumar, Aounon et al. "Certifying LLM Safety Against Adversarial Prompting",ICLR 2024 (2024)[链接](https://arxiv.org/abs/2309.02705)

[5]Emekci, Hakan, and Gülsüm Budakoglu. "Securing with Dual-LLM Architecture: ChatTEDU an Open Access Chatbot’s Defense",ISSS journal of micro and smart systems PP.99 (2025): 1-1.[链接](https://ieeexplore.ieee.org/document/11207588)

---





