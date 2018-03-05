import java.io.{BufferedWriter, FileWriter}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Matrices}

import scala.math.pow
import com.opencsv
import com.opencsv.CSVWriter

case class Params(bufsize: Int, maskcount: Int, framesize: Int, stacks: Int, stackheight: Int, repeats: Int)

object FastDotBench {
  var outputFile = new BufferedWriter(new FileWriter("results.csv"))
  var csvWriter = new CSVWriter(outputFile)

  // using a block as a kind-of context manager
  def time[R](block: => R): (R, Double) = {
    val t0 = System.nanoTime()
    val result : R = block    // call-by-name
    val t1 = System.nanoTime()
    val delta = (t1 - t0) / (1000.0 * 1000.0 * 1000.0)
    (result, delta)
  }

  def iter_dot(data: Seq[DenseMatrix], masks: DenseMatrix, params: Params): Unit = {
    for(i <- 1 to params.repeats) {
      for(dataChunk <- data) {
        dataChunk.multiply(masks.transpose)
      }
    }
  }

  def benchmark(params: Params): Double = {
    val masks = DenseMatrix.ones(params.maskcount, params.framesize)
    //val masks = DenseMatrix.ones(params.framesize, params.maskcount)

    // in the Python and C versions, data is a large contiguous buffer,
    // here we have a Seq of matrices. for cache effects this _should_ be
    // roughly equivalent.
    val data : Seq[DenseMatrix] = for (i <- 1 to params.stacks) yield {
      DenseMatrix.ones(params.stackheight, params.framesize)
    }
    iter_dot(data, masks, params.copy(repeats = 2))
    val (result, dt) = time {
      iter_dot(data, masks, params)
    }
    dt
  }

  def makeParams(): List[Params] = {
    val bufsize = 256 * 1024 * 1024
    val itemsize = 8 // TODO: automatically determine this?
    val maxmask = 16

    for {
      frameWidth <- List(256, 512, 1024, 2048, 4096)
      framesize : Int = pow(frameWidth, 2).intValue()
      frames : Int = bufsize / framesize / itemsize
      stackheight <- List(4, 8, 16, 32, 64, 128) if stackheight <= frames
      stacks : Int = frames / stackheight
      maskcount <- List(1, 2, 4, 8, 16)
    } yield Params(bufsize, maskcount, framesize, stacks, stackheight, repeats = maxmask / maskcount)
  }

  def writeResults(dt: Double, count: Int, params: Params) {
    csvWriter.writeNext(Array(count, params.bufsize, params.framesize, params.stackheight, params.framesize, params.maskcount, "", 0, "%.4f".format(dt), 0).map(x => x.toString))
    csvWriter.flush()
  }

  def main(args: Array[String]): Unit = {
    csvWriter.writeNext(Array("count", "bufsize", "framesize", "stackheight", "tilesize", "maskcount", "i", "blind", "hot", "hot-blind"))

    for((params, i) <- makeParams.zipWithIndex) {
      val dt = benchmark(params)
      writeResults(dt, i + 1, params)
    }
    csvWriter.close()
  }
}