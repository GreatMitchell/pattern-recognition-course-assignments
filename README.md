# 多模态行为识别系统 - MultiModal Action Recognition (MMAR)

基于MMAR数据集的多模态行为识别系统，实现了从传统特征方法到深度学习方法的完整流程。

## 📁 项目结构

```
MMAR_Project/
├── Baseline/                          # 第一阶段：传统方法
│   ├── config/                        # 数据路径配置
│   ├── features/                      # 特征提取模块
│   ├── models/                        # SVM分类器实现和多模态融合策略
│   ├── utils/                         # 工具函数
│   └── notebook_base.ipynb            # 完整流程和详细解释
├── data/                              # 数据集目录（需自行下载）
├── environment.yml                    # Conda环境配置
└── README.md
```

## 🚀 快速开始

### 1. 使用Conda配置环境

```bash
# 从environment.yml创建环境
conda env create -f environment.yml
conda activate mmar_project
```

### 2. 数据集准备

数据集可从以下任一仓库获取：
- **Gitee**: [Pattern_recognition_dataset_download](https://gitee.com/guqingxiang/Pattern_recognition_dataset_download/blob/main/README.md)
- **GitHub**: [Pattern_recognition_dataset_download](https://github.com/qingxiangjia/Pattern_recognition_dataset_download/blob/main/README.md)

下载后，请按如下结构组织数据集：
```
data/
│── train_500/
│   ├── rgb_data/
│   │   ├── 1/          # 行为类别1的RGB图像序列
│   │   ├── 2/          # 行为类别2的RGB图像序列
│   │   └── .../
│   ├── depth_data/     # 深度图像数据
│   │   ├── 1/          # 行为类别1的深度图像序列
│   │   ├── 2/          # 行为类别2的深度图像序列
│   │   └── .../
│   └── infrared_data/  # 红外图像数据（类似结构）
└── test_200/
    ├── rgb_data/
    │   ├── 1/          # 行为类别1的RGB图像序列
    │   ├── 2/          # 行为类别2的RGB图像序列
    │   └── .../
    ├── depth_data/     # 深度图像数据
    │   ├── 1/          # 行为类别1的深度图像序列
    │   ├── 2/          # 行为类别2的深度图像序列
    │   └── .../
    └── infrared_data/  # 红外图像数据（类似结构）
```

### 3. 运行代码

建议从Jupyter Notebook开始，运行[`Baseline\src\basic.ipynb`](Baseline\src\basic.ipynb)

```bash
jupyter notebook
```

## 📊 项目特性

### 第一阶段：传统方法基线（已完成 ✅）
- **特征提取**：HOG、LBP、SIFT等传统特征
- **分类器**：支持向量机（SVM）
- **融合策略**：
  - 早期融合（特征拼接）
  - 晚期融合（投票机制、加权融合）
- **性能**：在验证集上达到97.6%准确率（RGB模态）

### 第二阶段：深度学习方法（规划中 🔄）
- 多流CNN架构
- 注意力融合机制
- 跨模态学习

### 第三阶段：AI安全与鲁棒性（规划中 🔄）
- 模态缺失测试
- 对抗鲁棒性分析

## 🎯 核心算法

以下代码仅为示例。

### 特征提取
```python
# 支持的特征类型
features = {
    'HOG': extractor.extract_hog(image),
    'LBP': extractor.extract_lbp(image), 
    'SIFT': extractor.extract_sift(image)
}
```

### 多模态融合
```python
# 晚期融合 - 加权投票
fusion = WeightedVoteFusion(weights=[0.4, 0.25, 0.35])  # RGB, Depth, Infrared
result = fusion.fuse(rgb_pred, depth_pred, infrared_pred)
```

## 📈 实验最佳结果

| 模态 | 准确率 | 说明 |
|------|--------|------|
| RGB | 97.6% | 可见光模态表现最佳 |
| 红外 | 92.8% | 热成像模态表现良好 |
| 深度 | 82.4% | 深度信息单独使用有限 |
| 早期融合 | 92.8% | 特征层面融合 |
| 晚期融合 | 96.3% | 决策层面融合 |

## 🔧 配置说明

### 环境要求
- Python 3.9.24
- numpy 1.21.5
- scipy 1.10.1
- scikit-image 0.21.0
- scikit-learn 1.3.0
<!-- - PyTorch 1.9+ (第二阶段使用) -->

### 硬件建议
- CPU: 4核心以上
- 内存: 8GB以上
- 存储: 50GB可用空间（用于数据集）

## 🤝 协作指南

1. **代码规范**：遵循PEP8标准
2. **模块开发**：新功能请以模块形式添加到对应目录
3. **实验记录**：在notebook中记录重要实验结果
4. **分支管理**：新功能请在特性分支开发

## 📝 开发计划

- [x] 传统特征提取与SVM基线
- [x] 多模态融合策略实现
- [ ] 深度学习模型开发
- [ ] 模型鲁棒性测试
- [ ] 最终报告和可视化

## 📄 许可证

本项目仅供课程学习使用。

## 👥 贡献者

- 胡亦成 - 项目发起者和主要开发者

---

*如有问题，请创建Issue或联系项目维护者。*
