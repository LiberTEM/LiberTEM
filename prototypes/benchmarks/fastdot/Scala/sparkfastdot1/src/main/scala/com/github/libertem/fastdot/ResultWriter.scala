package com.github.libertem.fastdot

import java.io.{BufferedWriter, FileWriter}

import au.com.bytecode.opencsv.CSVWriter

class ResultWriter(fileName: String) {
  val outputFile = new BufferedWriter(new FileWriter(fileName))
  val csvWriter = new CSVWriter(outputFile)

  def writeHeader(extra: Seq[String] = List()): Unit = {
    csvWriter.writeNext(
      Array("count", "bufsize", "framesize", "stackheight", "tilesize", "maskcount", "i", "blind", "hot", "hot-blind")
      ++ extra
    )
  }

  def writeResults(dt: Double, count: Int, params: Params, extra: Seq[String] = List()) {
    csvWriter.writeNext(
      Array(count, params.bufsize, params.framesize, params.stackheight, params.framesize, params.maskcount, "", 0, "%.4f".format(dt), 0).map(x => x.toString)
      ++ extra
    )
    csvWriter.flush()
  }

  def withWriter[R](extra: Seq[String] = List())(fun: => R) : R = {
    writeHeader(extra)
    val result : R = fun
    close()
    result
  }

  def close(): Unit = {
    csvWriter.close()
  }
}
