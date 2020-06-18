package com.github.libertem.fastdot.converters

import java.io._
import java.nio.{ByteBuffer, ByteOrder, DoubleBuffer}

import scala.reflect.io.{File => ReflectFile}

object SplitBinary {
  val frameSize: Int = 128 * 128
  val rowSizeInBytes: Int = 128 * 4
  val frameSizeInBytes: Int = frameSize * 4
  val outPattern: String = "/tmp/scan_binary_files/%05d.raw"
  val in: String = "/tmp/scan_11_x256_y256.raw"
  val inputStream: DataInputStream = new DataInputStream(new FileInputStream(in))

  val numPartitions: Int = 32
  val numFrames: Int = (new File(in).length / (frameSizeInBytes + 2 * rowSizeInBytes)).toInt
  val framesPerPartition: Int = numFrames / numPartitions
  require(numFrames == 256 * 256, "numFrames is %d".format(numFrames))

  def readFrames(num: Int, out: Array[Float]) : Boolean = {
    val inputBytes: Array[Byte] = new Array[Byte](frameSizeInBytes)
    val inputByteBuffer = ByteBuffer.wrap(inputBytes)
    inputByteBuffer.order(ByteOrder.LITTLE_ENDIAN)
    val floatBuffer = inputByteBuffer.asFloatBuffer()
    for(frameNum <- 0 until num) {
      val bytesRead = inputStream.read(inputBytes, 0, frameSizeInBytes)
      require(bytesRead == -1 || bytesRead == frameSizeInBytes)
      floatBuffer.rewind()
      floatBuffer.get(out, frameNum * frameSize, frameSize)
      if(bytesRead == -1)
        return false // no more input data left
      inputStream.skipBytes(2 * rowSizeInBytes)
    }

    true
  }

  def main(args: Array[String]): Unit = {
    val partitionSizeBytes = framesPerPartition * frameSizeInBytes * 2 // why 2? float -> double
    val partitionByteArray = new Array[Byte](partitionSizeBytes)
    val partitionBuffer = ByteBuffer.wrap(partitionByteArray)

    // java is big endian by default. to interpret the bytes in the right order,
    // we need to change the bytebuffer to the native byte ordering on I/O
    partitionBuffer.order(ByteOrder.LITTLE_ENDIAN)

    val doubleBuffer = partitionBuffer.asDoubleBuffer()
    val partitionInputBuffer = new Array[Float](framesPerPartition * frameSize)
    var idx = 0

    while(readFrames(framesPerPartition, out = partitionInputBuffer)) {
      val out = outPattern.format(idx)
      val outStream = new DataOutputStream(ReflectFile(out).bufferedOutput())
      writeAsDoubles(partitionInputBuffer, doubleBuffer, partitionBuffer)
      outStream.write(partitionByteArray)
      outStream.close()
      idx += 1
    }
  }

  def writeAsDoubles(partition: Array[Float], doubleBuffer: DoubleBuffer, byteBuffer: ByteBuffer): Unit = {
    val doubles : Array[Double] = partition.map(_.doubleValue)
    println(doubles(0))
    doubleBuffer.rewind()
    byteBuffer.rewind()
    doubleBuffer.put(doubles)
  }
}
