package com.github.libertem.fastdot.converters

import java.io.{BufferedInputStream, DataInputStream, File, FileInputStream}
import java.nio.ByteBuffer
import java.util

import org.apache.avro.Schema
import org.apache.avro.file.DataFileWriter
import org.apache.avro.generic.{GenericData, GenericDatumWriter, GenericRecord}

import scala.collection.immutable.Stream
import collection.JavaConverters._
import collection.mutable._
import collection.immutable._

object BinaryToAvro {
  def main(args: Array[String]): Unit = {
    val in = "/tmp/scan_11_x256_y256.raw"
    val out = "/tmp/scan.avro"
    val schemaFile = "frame.avsc"
    val rowSize = 128 * 4
    val frameSize = 128 * rowSize

    val bis = new BufferedInputStream(new FileInputStream(in))
    val dis = new DataInputStream(bis)
    val frames = Stream.continually({
      val buf: Array[Byte] = new Array[Byte](frameSize)
      val bytesRead = dis.read(buf, 0, frameSize)
      val floatBuffer = ByteBuffer.wrap(buf).asFloatBuffer()
      val floats = new Array[Float](frameSize / 4)
      floatBuffer.get(floats)
      // the original frame size is 128x130, but the last two rows contain
      // garbage so we skip over them here
      dis.skipBytes(2 * rowSize)
      (floats, bytesRead)
    }).takeWhile(_._2 != -1).map(_._1)
    val schema : Schema = new Schema.Parser().parse(new File(schemaFile))
    val file = new File(out)
    val datumWriter = new GenericDatumWriter[GenericRecord](schema)
    val dataFileWriter = new DataFileWriter[GenericRecord](datumWriter)
    dataFileWriter.create(schema, file)
    for((frame, idx) <- frames.zipWithIndex) {
      val record : GenericRecord = new GenericData.Record(schema)
      val doubles: Array[Double] = frame.map(value => value.doubleValue())
      val javaDoubles : util.Collection[Double] = doubles.toIterable.asJavaCollection
      record.put("frame_data", javaDoubles)
      record.put("index", idx)
      dataFileWriter.append(record)
    }
    dataFileWriter.close()
  }
}
