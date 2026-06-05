package org.apache.spark.mllib.util

import java.io.File

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.classification.MultiLabeledPoint
import org.apache.spark.util.Utils

class MLXUtilsSuite extends SparkFunSuite with MLlibTestSparkContext {

  test("parse multi-label LibSVM records") {
    val (labels, indices, values) = MLXUtils.parseLibSVMRecord("0,2 1:1.0 3:2.0")
    assert(labels === Array(0.0, 2.0))
    assert(indices === Array(0, 2))
    assert(values === Array(1.0, 2.0))
  }

  test("load multi-label LibSVM file with inferred numFeatures") {
    val path = getTestResourcePath("multi_label_libsvm.txt")
    val points = MLXUtils.loadMultiLabelLibSVMFile(sc, path).collect()

    assert(points.length === 5)
    assert(points(0).labels === Array(0.0, 2.0))
    assert(points(0).features === Vectors.sparse(3, Seq((0, 1.0), (1, 0.5), (2, 1.0))))
    assert(points(1).labels === Array(2.0))
    assert(points(1).features === Vectors.sparse(3, Seq((1, 1.0))))
  }

  test("load multi-label LibSVM file with explicit numFeatures") {
    val path = getTestResourcePath("multi_label_libsvm.txt")
    val points = MLXUtils.loadMultiLabelLibSVMFile(sc, path, 4).collect()

    assert(points.forall(_.features.size === 4))
  }

  test("save and reload multi-label LibSVM file") {
    val examples = sc.parallelize(Seq(
      MultiLabeledPoint(Array(0.0, 1.0), Vectors.sparse(3, Seq((0, 1.0), (2, 2.0)))),
      MultiLabeledPoint(Array(2.0), Vectors.dense(0.5, 1.0, 0.0))
    ))

    val tempDir = Utils.createTempDir()
    val outputDir = new File(tempDir, "output")
    MLXUtils.saveAsMultiLabelLibSVMFile(examples, outputDir.getAbsolutePath)

    val loaded = MLXUtils.loadMultiLabelLibSVMFile(sc, outputDir.getAbsolutePath, 3).collect()
    assert(loaded.length === 2)
    assert(loaded(0).labels === Array(0.0, 1.0))
    assert(loaded(0).features === Vectors.sparse(3, Seq((0, 1.0), (2, 2.0))))
    assert(loaded(1).labels === Array(2.0))
    assert(loaded(1).features.toArray === Array(0.5, 1.0, 0.0))

    Utils.deleteRecursively(tempDir)
  }

  test("reject invalid feature format") {
    intercept[IllegalArgumentException] {
      MLXUtils.parseLibSVMRecord("1 1-1.0")
    }
  }

  test("reject unsorted feature indices") {
    intercept[IllegalArgumentException] {
      MLXUtils.parseLibSVMRecord("1 2:1.0 1:2.0")
    }
  }

  test("write LibSVM lines with comma-separated labels") {
    val examples = sc.parallelize(Seq(
      MultiLabeledPoint(Array(0.0, 2.0), Vectors.sparse(2, Seq((0, 1.0))))
    ))
    val tempDir = Utils.createTempDir()
    val outputDir = new File(tempDir, "output")
    MLXUtils.saveAsMultiLabelLibSVMFile(examples, outputDir.getAbsolutePath)

    val partFile = new File(outputDir.getAbsolutePath, "part-00000")
    assert(partFile.exists())
    val lines = scala.io.Source.fromFile(partFile)(scala.io.Codec.UTF8).getLines().toSet

    assert(lines === Set("0,2 1:1.0"))
    Utils.deleteRecursively(tempDir)
  }
}
