package org.apache.spark.ml.classification

import scala.collection.JavaConverters._
import scala.language.existentials

import java.util.UUID

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf, when, _}
import org.apache.spark.sql.types.{StructType, _}
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject, _}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.ml._
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair, Params}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

import java.util.{List => JList}

private[ml] trait MultiLabelOVRParams extends OneVsRestParams {

	setDefault(labelCol -> "labels")

	/** Validates and transforms the input schema. */
	override protected def validateAndTransformSchema(
		                                                 schema: StructType,
		                                                 fitting: Boolean,
		                                                 featuresDataType: DataType): StructType = {
		SchemaUtils.checkColumnType(schema, $(featuresCol), featuresDataType)
		val arrayDoubleType = new ArrayType(DoubleType, false)
		if (fitting) {
			val labelColName = $(labelCol)
			val labelDataType = schema(labelColName).dataType
			require(labelDataType == arrayDoubleType,
				s"The label column $labelColName must be array double type.")
		}
		SchemaUtils.appendColumn(schema, $(predictionCol), arrayDoubleType)
	}
}

final class MultiLabelOVRModel private[ml](
	                                          override val uid: String,
	                                          private[ml] val labelMetadata: Metadata,
	                                          val models: Array[_ <: ClassificationModel[_, _]])
	extends Model[MultiLabelOVRModel] with MultiLabelOVRParams with MLWritable {

	/** A Python-friendly auxiliary constructor. */
	private[ml] def this(uid: String, models: JList[_ <: ClassificationModel[_, _]]) = {
		this(uid, Metadata.empty, models.asScala.toArray)
	}

	def setFeaturesCol(value: String): this.type = set(featuresCol, value)

	def setPredictionCol(value: String): this.type = set(predictionCol, value)

	override def transformSchema(schema: StructType): StructType = {
		validateAndTransformSchema(schema, fitting = false, getClassifier.featuresDataType)
	}

	override def transform(dataset: Dataset[_]): DataFrame = {
		// Check schema
		transformSchema(dataset.schema, logging = true)

		// determine the input columns: these need to be passed through
		val origCols = dataset.schema.map(f => col(f.name))

		// add an accumulator column to store predictions of all the models
		val accColName = "mbc$acc" + UUID.randomUUID().toString
		val initUDF = udf { () => Map[Int, Double]() }
		val newDataset = dataset.withColumn(accColName, initUDF())

		// persist if underlying dataset is not persistent.
		val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
		if (handlePersistence) {
			newDataset.persist(StorageLevel.MEMORY_AND_DISK)
		}

		// update the accumulator column with the result of prediction of models
		val aggregatedDataset = models.zipWithIndex.foldLeft[DataFrame](newDataset) {
			case (df, (model, index)) =>
				val rawPredictionCol = model.getRawPredictionCol
				val columns = origCols ++ List(col(rawPredictionCol), col(accColName))

				// add temporary column to store intermediate scores and update
				val tmpColName = "mbc$tmp" + UUID.randomUUID().toString
				val updateUDF = udf { (predictions: Map[Int, Double], prediction: Vector) =>
					predictions + ((index, prediction(1)))
				}
				model.setFeaturesCol($(featuresCol))
				val transformedDataset = model.transform(df).select(columns: _*)
				val updatedDataset = transformedDataset
					.withColumn(tmpColName, updateUDF(col(accColName), col(rawPredictionCol)))
				val newColumns = origCols ++ List(col(tmpColName))

				// switch out the intermediate column with the accumulator column
				updatedDataset.select(newColumns: _*).withColumnRenamed(tmpColName, accColName)
		}

		if (handlePersistence) {
			newDataset.unpersist()
		}

		// output the index of the classifier with highest confidence as prediction
		val labelsUDF = udf { (predictions: Map[Int, Double]) =>
			predictions.filter(_._2 > 0.5)
				.map(_._1.toDouble)
  			.toArray
		}

		// output label and label metadata as prediction
		aggregatedDataset
			.withColumn($(predictionCol), labelsUDF(col(accColName)), labelMetadata)
			.drop(accColName)
	}

	override def copy(extra: ParamMap): MultiLabelOVRModel = {
		val copied = new MultiLabelOVRModel(
			uid, labelMetadata, models.map(_.copy(extra).asInstanceOf[ClassificationModel[_, _]]))
		copyValues(copied, extra).setParent(parent)
	}

	override def write: MLWriter = new MultiLabelOVRModel.OneVsRestModelWriter(this)
}

object MultiLabelOVRModel extends MLReadable[MultiLabelOVRModel] {

	override def read: MLReader[MultiLabelOVRModel] = new MultiLabelOVRModelReader

	override def load(path: String): MultiLabelOVRModel = super.load(path)

	/** [[MLWriter]] instance for [[MultiLabelOVRModel]] */
	private[MultiLabelOVRModel] class OneVsRestModelWriter(instance: MultiLabelOVRModel) extends MLWriter {

		OneVsRestParams.validateParams(instance)

		override protected def saveImpl(path: String): Unit = {
			val extraJson = ("labelMetadata" -> instance.labelMetadata.json) ~
				("numClasses" -> instance.models.length)
			OneVsRestParams.saveImpl(path, instance, sc, Some(extraJson))
			instance.models.zipWithIndex.foreach { case (model: MLWritable, idx) =>
				val modelPath = new Path(path, s"model_$idx").toString
				model.save(modelPath)
			}
		}
	}

	private class MultiLabelOVRModelReader extends MLReader[MultiLabelOVRModel] {

		/** Checked against metadata when loading model */
		private val className = classOf[MultiLabelOVRModel].getName

		override def load(path: String): MultiLabelOVRModel = {
			implicit val format = DefaultFormats
			val (metadata, classifier) = OneVsRestParams.loadImpl(path, sc, className)
			val labelMetadata = Metadata.fromJson((metadata.metadata \ "labelMetadata").extract[String])
			val numClasses = (metadata.metadata \ "numClasses").extract[Int]
			val models = Range(0, numClasses).toArray.map { idx =>
				val modelPath = new Path(path, s"model_$idx").toString
				DefaultParamsReader.loadParamsInstance[ClassificationModel[_, _]](modelPath, sc)
			}
			val ovrModel = new MultiLabelOVRModel(metadata.uid, labelMetadata, models)
			DefaultParamsReader.getAndSetParams(ovrModel, metadata)
			ovrModel.set("classifier", classifier)
			ovrModel
		}
	}

}

class MultiLabelOneVsRest(override val uid: String)
	extends Estimator[MultiLabelOVRModel] with MultiLabelOVRParams {

	def this() = this(Identifiable.randomUID("MultiLabelOVR"))

	def setClassifier(value: Classifier[_, _, _]): this.type = {
		set(classifier, value.asInstanceOf[ClassifierType])
	}

	def setLabelCol(value: String): this.type = set(labelCol, value)

	def setFeaturesCol(value: String): this.type = set(featuresCol, value)

	def setPredictionCol(value: String): this.type = set(predictionCol, value)

	override def transformSchema(schema: StructType): StructType = {
		validateAndTransformSchema(schema, fitting = true, getClassifier.featuresDataType)
	}

	override def fit(dataset: Dataset[_]): MultiLabelOVRModel = {
		transformSchema(dataset.schema)

		// determine number of classes either from metadata if provided, or via computation.
		val maxLabelUDF = udf { labels: Seq[Double] =>
			labels.max
		}
		val numClasses = dataset.agg {
			max(maxLabelUDF(col($(labelCol))).cast(DoubleType))
		}
			.head()
			.getAs[Double](0)
			.toInt

		val multiLabeled = dataset.select($(labelCol), $(featuresCol))

		// persist if underlying dataset is not persistent.
		val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
		if (handlePersistence) {
			multiLabeled.persist(StorageLevel.MEMORY_AND_DISK)
		}

		// create k columns, one for each binary classifier.
		val labelsContains: Int => UserDefinedFunction = (index: Int) => {
			udf { labels: Seq[Double] =>
				labels.toSet.contains(index.toDouble)
			}
		}
		val models = Range(0, numClasses).par.map { index =>
			// generate new label metadata for the binary problem.
			val newLabelMeta = BinaryAttribute.defaultAttr.withName("label").toMetadata()
			val labelColName = "mc2b$" + index

			val trainingDataset = multiLabeled.withColumn(
				labelColName, when(labelsContains(index)(col($(labelCol))), 1.0).otherwise(0.0), newLabelMeta)
			val classifier = getClassifier
			val paramMap = new ParamMap()
			paramMap.put(classifier.labelCol -> labelColName)
			paramMap.put(classifier.featuresCol -> getFeaturesCol)
			paramMap.put(classifier.predictionCol -> getPredictionCol)
			classifier.fit(trainingDataset, paramMap)
		}.toArray[ClassificationModel[_, _]]

		if (handlePersistence) {
			multiLabeled.unpersist()
		}

		// extract label metadata from label column if present, or create a nominal attribute
		// to output the number of labels
		val labelAttribute = NominalAttribute.defaultAttr.withName("label").withNumValues(numClasses)
		val model = new MultiLabelOVRModel(uid, labelAttribute.toMetadata(), models).setParent(this)
		copyValues(model)
	}

	override def copy(extra: ParamMap): MultiLabelOneVsRest = {
		val copied = defaultCopy(extra).asInstanceOf[MultiLabelOneVsRest]
		if (isDefined(classifier)) {
			copied.setClassifier($(classifier).copy(extra))
		}
		copied
	}
}

object MultiLabelOneVsRest extends MLReadable[MultiLabelOneVsRest] {

	override def read: MLReader[MultiLabelOneVsRest] = new MultiLabelOneVsRestReader

	override def load(path: String): MultiLabelOneVsRest = super.load(path)

	private[MultiLabelOneVsRest] class MultiLabelOneVsRestWriter(instance: MultiLabelOneVsRest) extends MLWriter {

		OneVsRestParams.validateParams(instance)

		override protected def saveImpl(path: String): Unit = {
			OneVsRestParams.saveImpl(path, instance, sc)
		}
	}

	private class MultiLabelOneVsRestReader extends MLReader[MultiLabelOneVsRest] {

		/** Checked against metadata when loading model */
		private val className = classOf[MultiLabelOneVsRest].getName

		override def load(path: String): MultiLabelOneVsRest = {
			val (metadata, classifier) = OneVsRestParams.loadImpl(path, sc, className)
			val ovr = new MultiLabelOneVsRest(metadata.uid)
			DefaultParamsReader.getAndSetParams(ovr, metadata)
			ovr.setClassifier(classifier)
		}
	}

}
