# 数学建模

本项目是 2024/7 学习 datawhale 数学建模课程项目的学习笔记和代码。


如果运行出现找不到包的情况，可以使用如下代码

```python
import os
import sys
 
cur_dir = os.getcwd()
root_dir = os.path.dirname(cur_dir)
sys.path.extend([cur_dir, root_dir])
# sys.path
```

## 项目结构

```tree
math-model
├─ design
│  └─ 7.1-AHP.excalidraw
├─ docs
│  ├─ 1-解析方法与几何模型.md
│  └─ 7-权重生成与评价模型.md
├─ LICENSE
├─ notebooks
│  ├─ data
│  │  └─ test.csv
│  ├─ imgs
│  │  └─ 第7章 权重生成与评价模型
│  │     ├─ image-1.png
│  │     └─ image.png
│  └─ 第7章 权重生成与评价模型.ipynb
├─ README.md
├─ requirements.txt
└─ src
   ├─ distance.py
   ├─ evaluation.py
   ├─ normalization.py
   └─ types.py

```

## 参考

- [数学建模课程](https://github.com/datawhalechina/intro-mathmodel/)
- [用人话讲明白AHP层次分析法（非常详细原理+简单工具实现）](https://blog.csdn.net/qq_41686130/article/details/122081827)
