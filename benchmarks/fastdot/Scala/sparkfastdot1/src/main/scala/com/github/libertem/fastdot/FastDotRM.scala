package com.github.libertem.fastdot

import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.SparkSession

/**
  * RowMatrix*DenseMatrix multiplication benchmark
  */
object FastDotRM {
  val sparkSession: SparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotRM").getOrCreate()


  /**
    * Eagerly multiply data with masks
    *
    * @param data the frame data as a distributed RowMatrix
    * @param masks the masks as a local DenseMatrix
    * @param params the benchmark parameters
    * @return a RowMatrix of the result
    */
  def iter_dot(data: RowMatrix, masks: DenseMatrix, params: Params): RowMatrix = {
    var result : RowMatrix = null
    for(i <- 1 to params.repeats) {
      // Multiply frame data with the masks. Under the hood, the dot product of the row vectors of the frames
      // with the columns of the masks is computed. Note that this is not cache efficient and gets slow as soon
      // as one frame and one mask don't fit into the L3 cache together anymore.
      // This computation is lazy, so after executing the following line, nothing is computed.
      result = data.multiply(masks)
      // An action to trigger computation. Without it, only the description how to do the computation
      // is created, the multiplication itself is not executed.
      // When this executes, tasks are sent to the cluster (or in this case, tasks are run on local threads)
      result.rows.foreach(row => row)
    }
    result
  }

  /**
    * Create a uniformly filled RowMatrix of the given size
    *
    * @param nRows number of rows
    * @param nCols number of columns
    * @param fill the value each entry of the matrix will have
    * @param slices number of slices, passed on to SparkContext.parallelize
    * @return a uniformly filled RowMatrix of the given size
    */
  def makeMatrix(nRows: Int, nCols: Int, fill : Double = 1.0, slices : Int): RowMatrix = {
    val rowVectors = for (_ <- 0L until nRows)
      yield new DenseVector(Array.fill(nCols)(fill))
    val matrixRdd = new RowMatrix(
      sparkSession.sparkContext.parallelize(rowVectors, numSlices = slices)
    )
    matrixRdd
  }

  /**
    * Benchmark matrix-matrix multiplication with the given parameters
    *
    * @param params the benchmark parameters
    * @return the time taken
    */
  def benchmark(params: Params): Double = {
    // Create the masks matrix as a local DenseMatrix
    // Under the hood, this creates a framesize*maskcount backing array that is filled with 1.0s
    val masks = DenseMatrix.ones(params.framesize, params.maskcount)

    // Make a data matrix that is sized and sliced according to the benchmark parameters
    // Note that this creates an RDD from data on the Spark Driver, which needs to be sent over with each task.
    val data = makeMatrix(nRows = params.frames.toInt, nCols = params.framesize, slices = params.stacks.toInt)

    // Execute the computation while measuring the time taken
    val (result, dt) = Utils.time {
      iter_dot(data, masks, params)
    }

    // Check result:
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
      for((params, i) <- ParamMaker.make(bufsizeMB = 256).zipWithIndex.reverse) {
        val dt = benchmark(params)
        writer.writeResults(dt, i + 1, params)
      }
    }
  }
}
