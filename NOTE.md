# 构建索引

核心参数：

* `DATA_PATH`: 基础数据 fvecs 格式.
* `KNNG_PATH`: kNN图路径.
* `L`: 控制NSG图的质量,越大越好.
* `R`: 控制图形的索引大小，最佳的R与数据集的内在维度有关.
* `C`: 构造NSG时的候选集最大大小。
* `NSG_PATH`: 生成的NSG索引.



## 图构造步骤


### load_data

功能：加载原始数据，分析数据集大小和数据维度。

### IndexNSG 构造函数

功能：初始化 `IndexNSG` 索引参数。

* `dimension`: 数据维度
* `n`: 数据集大小
* `m`: Metric类型，距离计算方式
* `initializer`: 初始化函数，默认为null

### build


### save


## 搜索步骤