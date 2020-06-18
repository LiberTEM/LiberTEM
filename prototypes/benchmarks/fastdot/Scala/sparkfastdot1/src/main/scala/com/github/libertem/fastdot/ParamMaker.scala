package com.github.libertem.fastdot
import scala.math.pow

case class Params(bufsize: Long, maskcount: Int, framesize: Int, stacks: Long, stackheight: Int, repeats: Int, frames: Long)

object ParamMaker {
  def make(bufsizeMB : Long = 256): List[Params] = {
    val bufsize : Long = bufsizeMB * 1024 * 1024
    val itemsize : Int = 8 // TODO: automatically determine this? java doesn't have sizeof
    val maxmask : Int = 16

    for {
      frameWidth <- List(256, 512, 1024, 2048, 4096)
      framesize : Int = pow(frameWidth, 2).intValue()
      frames : Long = bufsize / framesize / itemsize
      stackheight <- List(4, 8, 16, 32, 64, 128) if stackheight <= frames
      stacks : Long = frames / stackheight
      maskcount <- List(1, 2, 4, 8, 16)
    } yield Params(
      bufsize=bufsize,
      maskcount=maskcount,
      framesize=framesize,
      stacks=stacks,
      stackheight=stackheight,
      repeats=maxmask / maskcount,
      frames=frames
    )
  }
}