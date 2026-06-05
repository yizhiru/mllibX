package org.apache.spark.mllib.util

import java.io.{File, PrintWriter}
import java.net.URI

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.classification.MultiLabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.io.{Codec, Source}

object MLXUtils {

  private def formatLabel(label: Double): String = {
    val intLabel = label.toInt
    if (label == intLabel.toDouble) intLabel.toString else label.toString
  }

  private def isDistributedPath(path: String): Boolean = {
    if (!path.contains("://")) {
      false
    } else {
      Option(new URI(path).getScheme).getOrElse("file").toLowerCase != "file"
    }
  }

  private def resolveLocalPath(path: String): File = {
    if (path.contains("://")) new File(new URI(path)) else new File(path)
  }

  def loadMultiLabelLibSVMFile(
      sc: SparkContext,
      path: String): RDD[MultiLabeledPoint] = {
    loadMultiLabelLibSVMFile(sc, path, -1)
  }

  def loadMultiLabelLibSVMFile(
      sc: SparkContext,
      path: String,
      numFeatures: Int): RDD[MultiLabeledPoint] = {
    val parsed = parseMultiLibSVMFile(sc, path)

    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      val cached = parsed.persist(StorageLevel.MEMORY_ONLY)
      try {
        cached.map { case (_, indices, _) =>
          indices.lastOption.getOrElse(0)
        }.reduce(math.max) + 1
      } finally {
        cached.unpersist()
      }
    }

    parsed.map { case (labels, indices, values) =>
      MultiLabeledPoint(labels, Vectors.sparse(d, indices, values))
    }
  }

  private[spark] def parseMultiLibSVMFile(
      sc: SparkContext,
      path: String): RDD[(Array[Double], Array[Int], Array[Double])] = {
    readLines(sc, path)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map(parseLibSVMRecord)
  }

  private def readLocalLines(path: String): Seq[String] = {
    val file = resolveLocalPath(path)
    require(file.exists(), s"Path does not exist: $path")
    if (file.isDirectory) {
      file.listFiles()
        .filter(_.getName.startsWith("part-"))
        .sortBy(_.getName)
        .flatMap(f => Source.fromFile(f)(Codec.UTF8).getLines())
        .toSeq
    } else {
      Source.fromFile(file)(Codec.UTF8).getLines().toSeq
    }
  }

  private[spark] def readLines(sc: SparkContext, path: String): RDD[String] = {
    if (isDistributedPath(path)) {
      sc.textFile(path)
    } else {
      sc.parallelize(readLocalLines(path))
    }
  }

  private[spark] def parseLibSVMRecord(line: String): (Array[Double], Array[Int], Array[Double]) = {
    val items = line.split(' ')
    require(items.nonEmpty, s"LibSVM record must not be empty: line=$line")
    val labels = items.head
      .split(",")
      .filter(_.nonEmpty)
      .map(_.toDouble)
    val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
      val indexAndValue = item.split(':')
      require(indexAndValue.length == 2,
        s"Feature entry must be index:value; found $item in line=$line")
      val index = indexAndValue(0).toInt - 1
      val value = indexAndValue(1).toDouble
      (index, value)
    }.unzip

    var previous = -1
    var i = 0
    val indicesLength = indices.length
    while (i < indicesLength) {
      val current = indices(i)
      require(current > previous, s"indices should be one-based and in ascending order;" +
        s""" found current=$current, previous=$previous; line="$line"""")
      previous = current
      i += 1
    }
    (labels, indices, values)
  }

  def saveAsMultiLabelLibSVMFile(data: RDD[MultiLabeledPoint], dir: String): Unit = {
    val lines = data.map { mp =>
      val sb = new StringBuilder(mp.labels.map(formatLabel).mkString(","))
      mp.features.foreachActive { case (i, v) =>
        sb += ' '
        sb ++= s"${i + 1}:$v"
      }
      sb.toString()
    }
    if (isDistributedPath(dir)) {
      lines.saveAsTextFile(dir)
    } else {
      val outputDir = resolveLocalPath(dir)
      if (!outputDir.exists()) {
        require(outputDir.mkdirs(), s"Failed to create output directory: $dir")
      }
      val writer = new PrintWriter(new File(outputDir, "part-00000"))
      try {
        lines.collect().foreach(writer.println)
      } finally {
        writer.close()
      }
    }
  }

}
