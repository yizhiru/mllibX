/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.util

import java.io.{File, IOException}

import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.Dataset
import org.scalatest.Suite

trait DefaultReadWriteTest extends TempDirectory {
  self: Suite =>

  def testDefaultReadWrite[T <: Params with MLWritable](
      instance: T,
      testParams: Boolean = true): T = {
    val uid = instance.uid
    val subdirName = Identifiable.randomUID("test")

    val subdir = new File(tempDir, subdirName)
    val path = new File(subdir, uid).getPath

    instance.save(path)
    intercept[IOException] {
      instance.save(path)
    }
    instance.write.overwrite().save(path)
    val loader = instance.getClass.getMethod("read").invoke(null).asInstanceOf[MLReader[T]]
    val newInstance = loader.load(path)

    assert(newInstance.uid === instance.uid)
    if (testParams) {
      instance.params.foreach { p =>
        if (instance.isDefined(p)) {
          (instance.getOrDefault(p), newInstance.getOrDefault(p)) match {
            case (Array(values), Array(newValues)) =>
              assert(values === newValues, s"Values do not match on param ${p.name}.")
            case (value, newValue) =>
              assert(value === newValue, s"Values do not match on param ${p.name}.")
          }
        } else {
          assert(!newInstance.isDefined(p), s"Param ${p.name} shouldn't be defined.")
        }
      }
    }

    val load = instance.getClass.getMethod("load", classOf[String])
    val another = load.invoke(instance, path).asInstanceOf[T]
    assert(another.uid === instance.uid)
    another
  }

  def testEstimatorAndModelReadWrite[
      E <: Estimator[M] with MLWritable, M <: Model[M] with MLWritable](
      estimator: E,
      dataset: Dataset[_],
      testParams: Map[String, Any],
      checkModelData: (M, M) => Unit): Unit = {
    testParams.foreach { case (p, v) =>
      estimator.set(estimator.getParam(p), v)
    }
    val model = estimator.fit(dataset)

    val estimator2 = testDefaultReadWrite(estimator)
    testParams.foreach { case (p, v) =>
      val param = estimator.getParam(p)
      assert(estimator.get(param).get === estimator2.get(param).get)
    }

    val model2 = testDefaultReadWrite(model)
    testParams.foreach { case (p, v) =>
      val param = model.getParam(p)
      assert(model.get(param).get === model2.get(param).get)
    }

    checkModelData(model, model2)
  }
}
