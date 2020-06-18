package com.github.libertem.fastdot

import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession

object FastDotFromParquet {
  val sparkSession: SparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotFromParquet").master("local[4]").getOrCreate()

  def main(args: Array[String]): Unit = {
    //for(i <- 1 to 10) {
      doReadParquet()
    //}
  }

  def doReadParquet() : Unit = {
    val df = sparkSession.read.parquet("/tmp/scan_11_x256_y256.raw.parquet")
    df.printSchema()

    // later: read these from the input file
    val framesize = 128 * 128
    val frames = 256 * 256
    val maskcount = 8 // just picked one mask count, to keep it simple for now
    val rowVectors = df.rdd.map(
      row => Vectors.dense(row.getAs[Seq[Double]]("frames").toArray)
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
