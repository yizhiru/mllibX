package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.mllib.util.{MLXUtils, MLlibTestSparkContext}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{ArrayType, DoubleType}

class MultiLabelOneVsRestSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  import testImplicits._

  @transient var dataset: Dataset[_] = _

  test("multi-label OVR train") {
    dataset = MLXUtils.loadMultiLabelLibSVMFile(sc, "data/multi_label_libsvm.txt")
      .toDF()
    dataset.show()

    val numClasses = 20
    val ova = new MultiLabelOneVsRest()
      .setClassifier(new LogisticRegression)
    assert(ova.getLabelCol === "labels")
    assert(ova.getPredictionCol === "prediction")
    val ovaModel = ova.fit(dataset)

    // copied model must have the same parent.
    MLTestingUtils.checkCopy(ovaModel)

    assert(ovaModel.models.length === numClasses)
    val transformedDataset = ovaModel.transform(dataset)

    // check for label metadata in prediction col
    val predictionColSchema = transformedDataset.schema(ovaModel.getPredictionCol)
    assert(predictionColSchema.dataType == new ArrayType(DoubleType, false))
    transformedDataset.show()
  }
}
