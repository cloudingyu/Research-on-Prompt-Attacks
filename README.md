# LLM Prompt Injection Research

> **课程名称：** [信息安全导论]  
> **汇报时间：** 第 15-16 周  
> **最终报告提交：** 2026.01.09  
> **小组负责人：** [cloudingyu]

## 项目简介

本项目旨在调研大语言模型（LLM）面临的**提示词注入（Prompt Injection）**与**越狱（Jailbreaking）**安全威胁。我们将从技术发展现状出发，剖析 Transformer 架构下的漏洞原理，通过实战案例演示攻击效果，并最终提出并实现基于代码层面的防御方案。

## 项目结构说明 

本仓库采用严格的分层管理，请各位组员在提交时遵守以下目录结构：

```text
Project-Root/
├── README.md           # 项目说明文档
├── src/                # [最终] 整合后的核心演示代码
├── img/                # [最终] 用于 Word 报告和 PPT 的高清图表
│   ├── arch/           # 原理图、架构图 (Visio导出)
│   └── result/         # 实验结果截图、图表
├── cite/               # [最终] 参考文献库
│   ├── refs.bib        # BibTeX 格式引用
│   └── papers/         # 核心参考论文的 PDF 存档
├── doc/                # 项目任务书、markdown版论文、汇报文字稿
└── task/               # 组员个人的工作目录
    ├── 01_survey/      # [组员A] 负责综述：存放调研笔记、论文PDF、草稿
    ├── 02_principle/   # [组员B] 负责原理：存放 Visio 源文件、底层源码分析文档
    ├── 03_experiment/  # [组员C] 负责实验：存放测试用的 Prompts、原始截图、Excel数据
    └── 04_dev/         # [组员D] 负责代码：存放开发过程中的测试脚本、环境配置
```