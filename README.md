# mllibX

基于 Spark MLlib 的扩展库，面向多标签分类场景，补充标准 LibSVM 格式不支持多标签的问题。

## 功能

| 模块 | 说明 |
|------|------|
| `MLXUtils` | 多标签 LibSVM 文件读写（RDD API） |
| `MultiLabeledPoint` | 多标签样本数据结构 |
| `MultiLabelOneVsRest` | 多标签 One-vs-Rest 分类器（Spark ML Pipeline） |

## 环境要求

| 组件 | 版本 |
|------|------|
| Spark | 2.4.8（provided，需集群/运行时自行提供） |
| Scala | 2.12 |
| JDK（构建/测试） | 8 或 11（推荐） |

> JDK 18+ 上本地 LibSVM 读写已绕过 Hadoop FS；模型 `save`/`load` 仍依赖 Hadoop，在 JDK 18+ 环境下该能力不可用。

## 构建与测试

```bash
mvn test
```

测试数据位于 `src/test/resources/multi_label_libsvm.txt`。

## 数据格式

多标签 LibSVM 在标准 LibSVM 基础上，将行首单个标签扩展为逗号分隔的多个标签：

```
0,2 1:1.0 2:0.5 3:1.0
2 1:0.0 2:1.0 3:0.0
0,1 1:1.0 2:0.0 3:1.0
```

- 标签：逗号分隔的数值，如 `0,2` 表示样本同时属于标签 0 和 2
- 特征：与标准 LibSVM 相同，**1-based** 索引，升序排列，格式为 `index:value`

## 快速开始

### 加载数据

```scala
import org.apache.spark.mllib.util.MLXUtils

val data = MLXUtils.loadMultiLabelLibSVMFile(sc, "path/to/data.txt")
val df = data.toDF()  // 列名: labels (Array[Double]), features (Vector)
```

指定特征维度（可选，避免额外扫描）：

```scala
val data = MLXUtils.loadMultiLabelLibSVMFile(sc, "path/to/data.txt", numFeatures = 1000)
```

### 训练多标签分类器

```scala
import org.apache.spark.ml.classification.{LogisticRegression, MultiLabelOneVsRest}

val ovr = new MultiLabelOneVsRest()
  .setClassifier(new LogisticRegression())
  .setLabelCol("labels")
  .setFeaturesCol("features")
  .setThreshold(0.5)

val model = ovr.fit(df)
val predictions = model.transform(df)
```

### 保存数据

```scala
MLXUtils.saveAsMultiLabelLibSVMFile(data, "path/to/output")
```

本地路径直接写文件；HDFS 等分布式路径走 Spark 原生 `saveAsTextFile`。

## API 参考

### MultiLabelOneVsRest

继承 Spark `OneVsRest` 参数，额外提供：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `labelCol` | `labels` | 标签列，类型为 `Array[Double]` |
| `featuresCol` | `features` | 特征列 |
| `predictionCol` | `prediction` | 预测输出列，类型为 `Array[Double]` |
| `threshold` | `0.5` | 正类概率阈值，超过该值的标签纳入预测结果 |
| `classifier` | — | 底层二分类器（如 `LogisticRegression`） |

**训练逻辑**：根据数据集中最大标签值自动确定类别数（`maxLabel + 1`），为每个标签训练一个二分类器。

**预测逻辑**：对每个二分类器取正类概率（`rawPrediction` 第 2 维），超过 `threshold` 的标签索引组成最终预测数组。

### MLXUtils

```scala
// 加载（自动推断特征维度）
def loadMultiLabelLibSVMFile(sc: SparkContext, path: String): RDD[MultiLabeledPoint]

// 加载（指定特征维度）
def loadMultiLabelLibSVMFile(sc: SparkContext, path: String, numFeatures: Int): RDD[MultiLabeledPoint]

// 保存
def saveAsMultiLabelLibSVMFile(data: RDD[MultiLabeledPoint], dir: String): Unit
```

## 项目结构

```
src/main/scala/org/apache/spark/
├── ml/classification/
│   └── MultiLabelOneVsRest.scala    # Estimator / Model
├── mllib/classification/
│   └── MultiLabeledPoint.scala      # 数据类
└── mllib/util/
    └── MLXUtils.scala               # LibSVM 读写工具
```

## 限制与说明

- 当前仅适配 **Spark 2.4.8**，不同 Spark 版本内部 API 差异较大，升级需逐一验证。
- 包名使用 `org.apache.spark.*`，部署时需与 Spark 放在同一 classpath。
- 标签约定为 **0-based 数值索引**（0, 1, 2, …），训练时按 `max(labels) + 1` 确定分类器数量。
