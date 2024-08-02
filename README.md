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
├─ docs 笔记
├─ LICENSE
├─ notebooks 案例代码
│  └─ 第7章 权重生成与评价模型.ipynb
├─ README.md
├─ requirements.txt
└─ src 主要代码

```

## 参考

- [数学建模课程](https://github.com/datawhalechina/intro-mathmodel/)
- 其余参考在相应笔记中列出
