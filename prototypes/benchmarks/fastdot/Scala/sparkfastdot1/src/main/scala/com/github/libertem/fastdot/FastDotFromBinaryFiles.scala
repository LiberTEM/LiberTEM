package com.github.libertem.fastdot

import java.io.{DataInputStream, File}
import java.nio.{ByteBuffer, ByteOrder}

import org.apache.spark.SparkContext
import org.apache.spark.input.PortableDataStream
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.sql.SparkSession

object FastDotFromBinaryFiles {
  val sparkSession: SparkSession = Utils.setupSparkSession(withLogging = true).appName("FastDotOnBinaryFiles").master("local[8]").getOrCreate()
  val sc: SparkContext = sparkSession.sparkContext
  val elemSize = 8
  val frameSize: Int = 128 * 128
  val maskCount = 8 // just picked one mask count, to keep it simple for now
  var numFrames: Int = 2048
  val fileSize: Int = 128 * 128 * elemSize * numFrames
  val frames: Int = 256 * 256
  val stackHeight = 32

  def main(args: Array[String]): Unit = {
    for(i <- 1 to 30) {
      readInStacks()
    }
  }

  def readInStacks(): Unit = {
    val (totalBytesProcessed, dt) = Utils.time {
      val masks = DenseMatrix.ones(frameSize, maskCount)
      sc.binaryFiles("/tmp/scan_binary_files/*").map({ case (filename: String, inputStream: PortableDataStream) =>
        val input: DataInputStream = inputStream.open()

        val bytes = new Array[Byte](stackHeight * frameSize * elemSize)
        val doubles = new Array[Double](frameSize * stackHeight)

        var bytesProcessed: Long = 0L

        try {
          // FIXME: possibility of short reads here?
          // I think they can happen when we are interrupted by a signal while reading
          while (input.read(bytes) != -1) {
            bytesProcessed += bytes.length
            val byteBuffer = ByteBuffer.wrap(bytes)
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val doubleBuffer = byteBuffer.asDoubleBuffer()

            while (doubleBuffer.hasRemaining) {
              doubleBuffer.get(doubles)
              val dataMatrix: DenseMatrix = new DenseMatrix(
                numCols = frameSize,
                numRows = stackHeight,
                values = doubles
              )
              val result: DenseMatrix = dataMatrix.multiply(masks)
            }
          }
        } finally {
          input.close()
        }
        bytesProcessed
      }).reduce(_ + _)
    }
    println("readInStacks finished processing %d bytes in %.3f seconds".format(totalBytesProcessed, dt))
  }
}
