package com.github.libertem.fastdot

import java.io.File

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import com.databricks.spark.avro._
import org.apache.avro.Schema

object FastDotFromAvro {
  def main(args: Array[String]): Unit = {
    val sparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotFromAvro").master("local[4]").getOrCreate()

    //val df = sparkSession.read.parquet("/tmp/scan_11_x256_y256.raw.parquet")
    val schema = new Schema.Parser().parse(new File("frame.avsc"))

    val df = sparkSession.read
      .option("avroSchema", schema.toString)
      .avro("/tmp/scan.avro")
    df.printSchema()

    // later: read these from the input file
    val framesize = 128 * 128
    val frames = 256 * 256
    val maskcount = 8 // just picked one mask count, to keep it simple for now
    val rowVectors = df.rdd.map(
      row => Vectors.dense(row.getAs[Seq[Double]]("frame_data").toArray)
    )
    val data = new RowMatrix(rowVectors)
    val masks = DenseMatrix.ones(framesize, maskcount)
    val (result, dt) = Utils.time {
      val result = data.multiply(masks)
      // action to 'start' the calculation:
      result.rows.foreach(row => row)
      result
    }
    println("%.3f".format(dt))
  }
}
