package org.apache.spark.mllib.util

import scala.util.Properties

object JdkCompat {

  /** Hadoop FileSystem APIs break on JDK 18+ due to Subject.getSubject removal. */
  def supportsHadoopFileSystem: Boolean = {
    val version = Properties.javaVersion
    val major = version.split("[^0-9]").filter(_.nonEmpty).headOption.map(_.toInt).getOrElse(8)
    major < 18
  }
}
