package org.apache.spark.mllib.classification

import org.apache.spark.ml.linalg.Vector

import scala.beans.BeanInfo

@BeanInfo
case class MultiLabeledPoint(labels: Array[Double], features: Vector) {
  override def toString: String = {
    s"(${labels.mkString(",")},$features)"
  }
}
