package org.apache.spark.mllib.util

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.classification.MultiLabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object MLXUtils {

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

		// Determine number of features.
		val d = if (numFeatures > 0) {
			numFeatures
		} else {
			parsed.persist(StorageLevel.MEMORY_ONLY)
			parsed.map { case (labels, indices, values) =>
				indices.lastOption.getOrElse(0)
			}.reduce(math.max) + 1
		}

		parsed.map { case (labels, indices, values) =>
			MultiLabeledPoint(labels, Vectors.sparse(d, indices, values))
		}
	}

	private[spark] def parseMultiLibSVMFile(
		                                       sc: SparkContext,
		                                       path: String): RDD[(Array[Double], Array[Int], Array[Double])] = {
		sc.textFile(path)
			.map(_.trim)
			.filter(line => !(line.isEmpty || line.startsWith("#")))
			.map(parseLibSVMRecord)
	}

	private[spark] def parseLibSVMRecord(line: String): (Array[Double], Array[Int], Array[Double]) = {
		val items = line.split(' ')
		val labels = items.head
			.split(",")
			.map(item => item.toDouble)
		val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
			val indexAndValue = item.split(':')
			// Convert 1-based indices to 0-based.
			val index = indexAndValue(0).toInt - 1
			val value = indexAndValue(1).toDouble
			(index, value)
		}.unzip

		// check if indices are one-based and in ascending order
		var previous = -1
		var i = 0
		val indicesLength = indices.length
		while (i < indicesLength) {
			val current = indices(i)
			require(current > previous, s"indices should be one-based and in ascending order;"
				+
				s""" found current=$current, previous=$previous; line="$line"""")
			previous = current
			i += 1
		}
		(labels, indices, values)
	}

	def saveAsMultiLabelLibSVMFile(data: RDD[MultiLabeledPoint], dir: String): Unit = {
		val dataStr = data.map { mp =>
			val sb = new StringBuilder(mp.labels.mkString(","))
			mp.features.foreachActive { case (i, v) =>
				sb += ' '
				sb ++= s"${i + 1}:$v"
			}
			sb.mkString
		}
		dataStr.saveAsTextFile(dir)
	}

}
