package com.github.libertem.fastdot

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._


/**
  * BlockMatrix*BlockMatrix multiplication benchmark
  */
object FastDotBM {
  val sparkSession: SparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotBMFromRDD").getOrCreate()

  def iter_dot(data: BlockMatrix, masks: BlockMatrix, params: Params): Unit = {
    for(i <- 1 to params.repeats) {
      // FIXME it may be possible that repeating here does not have the desired effect
      // because of caching (why? because calling iter_dot twice resulted in using cached values on second call,
      // judging by execution speed)
      // TODO: is this still affected by lazyness?
      var results = data.multiply(masks)
    }
  }

  def makeMatrix(nRows: Int, nCols: Int, fill : Double = 1.0, slices : Int): IndexedRowMatrix = {
    val rowVectors = for (_ <- 0L until nRows)
      yield new DenseVector(Array.fill(nCols)(fill))
    val indexedRows : Seq[IndexedRow] = rowVectors .zipWithIndex.map((row: (DenseVector, Int)) => row match {
      case (vector, index) => IndexedRow(index, vector)
    })
    val matrixRdd : IndexedRowMatrix = new IndexedRowMatrix(
      sparkSession.sparkContext.parallelize(indexedRows, numSlices = slices)
    )
    matrixRdd
  }

  case class BenchResults(dt: Double, dt_data: Double, dt_masks: Double)

  def benchmark(params: Params): BenchResults = {
    // TODO: parameters for block matrix size
    val (dataMatrix, dt_data) = Utils.time {
      makeMatrix(nRows = params.frames.toInt, nCols = params.framesize, slices = params.frames.toInt).toBlockMatrix()
    }
    var (masksMatrix, dt_masks) = Utils.time {
      makeMatrix(nRows = params.framesize, nCols = params.maskcount, slices = params.frames.toInt).toBlockMatrix()
    }

    //XXX in this case, we don't want to pre-warm the caches, as that can falsify the results
    //iter_dot(dataMatrix, masksMatrix, params.copy(repeats = 1))
    val (result, dt) = Utils.time {
      iter_dot(dataMatrix, masksMatrix, params)
    }
    BenchResults(dt=dt, dt_data=dt_data, dt_masks=dt_masks)
  }

  def main(args: Array[String]): Unit = {
    Utils.setupLogging()

    val writer = new ResultWriter("results-bm-rdd.csv")
    writer.withWriter(extra=List("make-data-matrix", "make-masks-matrix")) {
      for((params, i) <- ParamMaker.make().zipWithIndex) {
        val res = benchmark(params)
        writer.writeResults(res.dt, i + 1, params, extra=List(
          res.dt_data.toString,
          res.dt_masks.toString
        ))
      }
    }
  }
}
