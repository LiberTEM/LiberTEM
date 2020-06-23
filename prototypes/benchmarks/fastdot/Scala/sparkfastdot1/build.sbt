name := "sparkfastdot1"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= {
  val sparkVer = "2.3.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % "compile" withSources(),
    "org.apache.spark" %% "spark-mllib" % sparkVer
  )
}

libraryDependencies += "com.opencsv" % "opencsv" % "4.1"

// native libraries for netlib and breeze
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"


// FIXME: nd4j depends on a newer guava version than hadoop can handle, yay!
// val nd4jVersion = "0.7.2"
// libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion
// libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion

libraryDependencies += "com.databricks" %% "spark-avro" % "4.0.0"
