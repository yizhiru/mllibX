package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.mllib.util.{JdkCompat, MLXUtils, MLlibTestSparkContext}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{ArrayType, DoubleType}
import org.scalatest.Tag

class MultiLabelOneVsRestSuite extends SparkFunSuite
  with MLlibTestSparkContext
  with DefaultReadWriteTest {

  import testImplicits._

  @transient private var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val path = getTestResourcePath("multi_label_libsvm.txt")
    dataset = MLXUtils.loadMultiLabelLibSVMFile(sc, path).toDF()
  }

  test("fit and transform multi-label OVR model") {
    val ovr = new MultiLabelOneVsRest()
      .setClassifier(new LogisticRegression())
    assert(ovr.getLabelCol === "labels")
    assert(ovr.getPredictionCol === "prediction")
    assert(ovr.getThreshold === 0.5)

    val model = ovr.fit(dataset)
    MLTestingUtils.checkCopy(model)

    // max label is 2, so we need classifiers for labels 0, 1, and 2
    assert(model.models.length === 3)

    val transformed = model.transform(dataset)
    val predictionColSchema = transformed.schema(model.getPredictionCol)
    assert(predictionColSchema.dataType == new ArrayType(DoubleType, false))
    assert(transformed.count() === dataset.count())
  }

  test("numClasses includes the maximum label index") {
    val labels = Seq(
      (Array(0.0, 2.0), org.apache.spark.ml.linalg.Vectors.dense(1.0, 0.0, 1.0)),
      (Array(2.0), org.apache.spark.ml.linalg.Vectors.dense(0.0, 1.0, 0.0))
    ).toDF("labels", "features")

    val model = new MultiLabelOneVsRest()
      .setClassifier(new LogisticRegression())
      .fit(labels)

    assert(model.models.length === 3)
  }

  test("threshold param filters predicted labels") {
    val classifier = new LogisticRegression()
    val baseModel = new MultiLabelOneVsRest()
      .setClassifier(classifier)
      .setThreshold(0.5)
      .fit(dataset)

    val strictModel = baseModel.setThreshold(0.99)

    def labelCount(model: MultiLabelOVRModel): Int = {
      model.transform(dataset)
        .select("prediction")
        .collect()
        .map(_.getAs[Seq[Double]](0).length)
        .sum
    }

    assert(labelCount(strictModel) <= labelCount(baseModel))
  }

  object HadoopFsTag extends Tag("HadoopFs")

  test("estimator and model read/write", HadoopFsTag) {
    assume(JdkCompat.supportsHadoopFileSystem,
      "Model persistence requires JDK 8-17 (Hadoop FS is incompatible with JDK 18+)")
    val estimator = new MultiLabelOneVsRest()
      .setClassifier(new LogisticRegression())
      .setThreshold(0.4)
    testEstimatorAndModelReadWrite[MultiLabelOneVsRest, MultiLabelOVRModel](
      estimator,
      dataset,
      Map("threshold" -> 0.4),
      (model: MultiLabelOVRModel, model2: MultiLabelOVRModel) =>
        assert(model.models.length === model2.models.length))
  }
}
