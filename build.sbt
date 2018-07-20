name := "Test"

version := "0.1"

scalaVersion := "2.12.6"

libraryDependencies ++= Seq(
  "org.platanios" %% "tensorflow" % "0.2.4",
  "org.platanios" %% "tensorflow-data" % "0.2.4",
  "org.platanios" %% "tensorflow-api" % "0.2.4",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.0"
  //"org.platanios" %% "tensorflow" % "0.2.3" classifier "linux-cpu-x86_64"
)
        