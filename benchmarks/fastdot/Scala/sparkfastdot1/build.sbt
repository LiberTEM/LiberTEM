name := "sparkfastdot1"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= {
  val sparkVer = "2.2.1"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % "compile" withSources(),
    "org.apache.spark" %% "spark-mllib" % sparkVer
  )
}

libraryDependencies += "com.opencsv" % "opencsv" % "4.1"

// native libraries for netlib and breeze
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"
