# 构建索引

核心参数：

* `DATA_PATH`: 基础数据 fvecs 格式.
* `KNNG_PATH`: kNN图路径.
* `L`: 控制NSG图的质量,越大越好.
* `R`: 控制图形的索引大小，最佳的R与数据集的内在维度有关.
* `C`: 构造NSG时的候选集最大大小。
* `NSG_PATH`: 生成的NSG索引.

IndexNSG 私有成员：
final_graph_：最终生成的NSG图，顶点i的所有的近邻id
ep_：导航点 id（Neighbor对象的id）


## 图构造步骤


### load_data

功能：加载原始数据，分析数据集大小和数据维度。

原始数据 data_ 的结构？？？
NSG 的数据结构：点id【unsigned类型】，点坐标，邻居id；

### IndexNSG 构造函数

功能：初始化 `IndexNSG` 索引参数。

* `Index.dimension_ = dimension`: 数据维度
* `Index.nd_ = n`: 数据集大小
* `Index.distance_ = m`: Metric类型，距离计算方式
* `initializer`: 初始化函数，默认为null

### build

1. `Load_nn_graph`：加载 nn_graph 数据文件，存储到 `IndexNSG.final_graph_`。
2. `init_graph`初始化nsg图，获取**导航点** - 距离数据集质心最近的点为导航点id `IndexNSG.ep_`。
    
3. `Link`：建立顶点连边，构造裁剪后的图。
        按顶点id，多线程完成此步骤：
        1. 获取knn 图中的每个顶点的邻居节点
        2. 对每个顶点的邻居节点按策略裁边，并构造结果图。

        裁边策略：加入候选集合的点 p满足：pq的距离 是 p与其他所有候选集合点中距离的最小值。
4. 依据连边数据，压缩最终的图索引结构。
5. `tree_grow`： 检查并将所有顶点是否加入图。
6. 统计出度。


#### 核心函数：

`get_neighbors` 获取query点的最近邻，待选集合个数为L 【TBD】
    参数：
      * query：查询
      * parameter：
      * flags：
      * retset：
      * fullset：



`tree_grow(parameters)`: TBD
从导航点开始，进行DFS 遍历，
1. 获取未加入到NSG图的顶点 p 和未加入的顶点总个数；
2. 将未加入到NSG图的顶点，依次加入到NSG图，
3. 以将p加入到NDG图的顶点作为新的root，循环重复步骤1；



### save

1. 保存NSG图的最大出度
2. 保存导航顶点。搜索的时候也从导航点开始搜索。
3. 保存每个顶点的邻居点个数和详细的邻居点id。


## 搜索步骤


贪心算法： 从导航点开始，选择邻居点中与query 最近的点为下次搜索的点，直到无法更新topk。

1. 初始化待选集合：导航点的所有邻居点，不足L的随机填充；
2. 选择topk：
        1. 初始化待选集合的点与query点的距离，并从小到大排序；
        2. 遍历待选集合中每个点，设置flag：
                1. 点的邻居与query的距离在待选集合中为前L短，则将邻居点加入到待选集合；
                2. 每处理完一个待选集合的点，则继续下一个点：
                        1. 要么，上一次搜索邻居点，使得点的距离离query更近，则选择该邻居点继续搜索；
                        2. 否则，挨个遍历待选集合。


## 优化图

优化的结果：保证NSG的每个顶点最多只有k个邻居。

### 问题

如何优化的图？？？？优化图的时候，norm方法返回的是一个float数据？


## 优化搜索
