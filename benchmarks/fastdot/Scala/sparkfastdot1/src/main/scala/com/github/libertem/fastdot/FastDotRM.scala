package com.github.libertem.fastdot

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.SparkSession

/**
  * RowMatrix*DenseMatrix multiplication benchmark
  */
object FastDotRM {
  val sparkSession: SparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotRM").getOrCreate()

  def iter_dot(data: RowMatrix, masks: DenseMatrix, params: Params): RowMatrix = {
    var result : RowMatrix = null
    for(i <- 1 to params.repeats) {
      result = data.multiply(masks)
      // an action to trigger computation:
      result.rows.foreach(row => row)
    }
    result
  }

  def makeMatrix(nRows: Int, nCols: Int, fill : Double = 1.0, slices : Int): RowMatrix = {
    val rowVectors = for (_ <- 0L until nRows)
      yield new DenseVector(Array.fill(nCols)(fill))
    val matrixRdd = new RowMatrix(
      sparkSession.sparkContext.parallelize(rowVectors, numSlices = slices)
    )
    matrixRdd
  }

  def benchmark(params: Params): Double = {
    val masks = DenseMatrix.ones(params.framesize, params.maskcount)
    val data = makeMatrix(nRows = params.frames.toInt, nCols = params.framesize, slices = params.stacks.toInt)

    val (result, dt) = Utils.time {
      val result = iter_dot(data, masks, params)
      result
    }

    // check result:
    val result_summed = result.rows.map(row => row.toArray.sum).sum()
    val expected = (params.frames * params.framesize * params.maskcount).toDouble
    assert(
       result_summed == expected,
      "%.2f != %.2f".format(result_summed, expected)
    )
    dt
  }

  def main(args: Array[String]): Unit = {
    Utils.setupLogging()
    val writer = new ResultWriter("results-tiled-irm-rdd.csv")
    writer.withWriter() {
      for((params, i) <- ParamMaker.make().zipWithIndex.reverse) {
        val dt = benchmark(params)
        writer.writeResults(dt, i + 1, params)
      }
    }
  }
}
