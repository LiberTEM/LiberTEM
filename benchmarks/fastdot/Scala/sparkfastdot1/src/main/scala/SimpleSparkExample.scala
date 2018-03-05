import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object SimpleSparkExample {
  def main(args: Array[String]): Unit = {
    val logFile = "/home/clausen/source/dask/README.rst";
    val conf = new SparkConf().setAppName("whateverapp").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val logData = sc.textFile(logFile, 2).cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}
