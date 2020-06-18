package com.github.libertem.fastdot

import org.apache.spark.mllib.linalg.DenseMatrix


/**
  * this is the baseline performance we can get in scala using local*local dense matrix multiplication
  */
object FastDotBench {
  def iter_dot(data: Seq[DenseMatrix], masks: DenseMatrix, params: Params): Unit = {
    for(i <- 1 to params.repeats) {
      for(dataChunk <- data) {
        dataChunk.multiply(masks.transpose)
      }
    }
  }

  def benchmark(params: Params): Double = {
    val masks = DenseMatrix.ones(params.maskcount, params.framesize)
    // with non-native BLAS, transposed vs non-transposed multiplication makes a difference:
    //val masks = DenseMatrix.ones(params.framesize, params.maskcount)

    // in the Python and C versions, data is a large contiguous buffer,
    // here we have a Seq of matrices. for cache effects this _should_ be
    // roughly equivalent.
    val data : Seq[DenseMatrix] = for (i <- 1L to params.stacks) yield {
      DenseMatrix.ones(params.stackheight, params.framesize)
    }
    // pre-warm the cache:
    iter_dot(data, masks, params.copy(repeats = 1))
    val (result, dt) = Utils.time {
      iter_dot(data, masks, params)
    }
    dt
  }

  def main(args: Array[String]): Unit = {
    val writer = new ResultWriter("results.csv")
    writer.withWriter() {
      for((params, i) <- ParamMaker.make().zipWithIndex) {
        val dt = benchmark(params)
        writer.writeResults(dt, i + 1, params)
      }
    }
  }
}