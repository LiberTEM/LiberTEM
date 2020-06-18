package com.github.libertem.fastdot

import org.apache.spark.sql.SparkSession

object Utils {
  // using a block as a kind-of context manager
  def time[R](block: => R): (R, Double) = {
    val t0 = System.nanoTime()
    val result : R = block    // call-by-name
    val t1 = System.nanoTime()
    val delta = (t1 - t0) / (1000.0 * 1000.0 * 1000.0)
    (result, delta)
  }

  def setupLogging(): Unit = {
    import org.apache.log4j.{Level, Logger}
    Logger.getLogger("com").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.OFF)
  }

  def setupSparkSession(withLogging : Boolean = false) : SparkSession.Builder = {
    var builder = SparkSession.builder()
      .master("local[4]")

    if(withLogging) {
      builder = builder.config("spark.eventLog.enabled", value = true)
        .config("spark.eventLog.dir", "file:/tmp/spark-events") // this is where the history server expects the event log(s)
    }
    builder
  }
}
