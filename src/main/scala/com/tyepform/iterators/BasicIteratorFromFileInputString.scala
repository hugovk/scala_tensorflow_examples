package com.tyepform.iterators

import java.io.File

import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.{Shape, Tensor}
import org.slf4j.LoggerFactory
import org.platanios.tensorflow.api._
import scala.util.{Try,Success,Failure}
import scala.util.Random

class BasicIteratorFromFileInputString(parmas:Map[String,String]){

  type parseInput = (Iterator[String],Iterator[String],List[List[Float]],List[Float]) =>
    (List[List[Float]],List[Float],Iterator[String],Iterator[String])

  val logger = Logger(LoggerFactory.getLogger(getClass.getName))
  val filesFolder = parmas("filesFolder")//"/Users/felipemateos/work/DataForTesting/regressionInTensorFlow"
  val inputsFile = s"$filesFolder${File.separator}${parmas("inputsFile")}"//s"$filesFolder${File.separator}inputs.tab"
  val labelsFile = s"$filesFolder${File.separator}${parmas("labelsFile")}"// s"$filesFolder${File.separator}labels.tab"
  val separator = parmas("separator")//"\\t"
  val batchSize = parmas("batchSize").toDouble//128
  val inputDimensionExpected = parmas("inputDimensionExpected").toInt //3
  val outputDimensionExpected = parmas("outputDimensionExpected").toInt
  private var inputIterartor : Iterator[String] = scala.io.Source.fromFile(inputsFile).getLines()
  private var labelsIterartor : Iterator[String] = scala.io.Source.fromFile(labelsFile).getLines()
  var epoch = 0.0
  private var numberOferrorsNonNA = 0
  val r = new Random()
  val cut : Double = parmas("cut").toDouble //0.5

  /**
    *
    * @param x
    * @param y
    * @param xs
    * @param ys
    * @return
    */
  def parseInput(x:Iterator[String],
                 y:Iterator[String],
                 xs:List[List[Float]],
                 ys:List[Float],linesRead:Int): ( Iterator[String], Iterator[String], List[List[Float]], List[Float],Int)={


    val dataPointFormated : Try[List[Float]] = Try( x.next().split(separator).map(_.trim().toFloat).toList )
    val labelPointFormated : Try[Float]      = Try( y.next().toFloat)

    (dataPointFormated,labelPointFormated) match {
      case (Success(i),Success(j)) =>
        if(r.nextFloat() > cut)  (x , y, i :: xs ,j :: ys ,linesRead + 1  )
        else   (x,y,xs,ys,linesRead)
      case (Failure(i),_) =>
        i match {
          case i:NumberFormatException =>
            //logger.error(s"${i.getMessage}")
            //logger.error(s"${linesRead}")
            if(i.getMessage.contains("NA")){
              (x,y,xs,ys,linesRead)
            }
            else {
              numberOferrorsNonNA +=1
              (x,y,xs,ys,linesRead)
            }
          case _ =>
            epoch += 1.0
            inputIterartor = scala.io.Source.fromFile(inputsFile).getLines()
            labelsIterartor = scala.io.Source.fromFile(labelsFile).getLines()
            (inputIterartor,labelsIterartor,xs,ys,linesRead + 1 )
        }
      case (_,Failure(i)) =>
        i match {
          case i:NumberFormatException =>
            //logger.error(s"${i.getMessage}")
            //logger.error(s"${linesRead}")
            if(i.getMessage.contains("NA")){
              (x,y,xs,ys,linesRead )
            }
            else {
              numberOferrorsNonNA +=1
              (x,y,xs,ys,linesRead )
            }
          case _ =>
            epoch += 1.0
            inputIterartor = scala.io.Source.fromFile(inputsFile).getLines()
            labelsIterartor = scala.io.Source.fromFile(labelsFile).getLines()
            (inputIterartor,labelsIterartor,xs,ys,linesRead + 1 )
        }
      case (Failure(j),Failure(i)) =>
        (i,j) match {
          case (i:NumberFormatException, j:NumberFormatException) =>
            //logger.error(s"${i.getMessage}")
            //logger.error(s"${linesRead}")
            if(i.getMessage.contains("NA")){
              (x,y,xs,ys,linesRead )
            }
            else {
              numberOferrorsNonNA +=1
              (x,y,xs,ys,linesRead )
            }
          case (_,_) =>
            epoch += 1.0
            inputIterartor = scala.io.Source.fromFile(inputsFile).getLines()
            labelsIterartor = scala.io.Source.fromFile(labelsFile).getLines()
            (inputIterartor,labelsIterartor,xs,ys,linesRead + 1 )
        }
    }
  }

  /**
    *
    * @return A batch for a regression problem
    */
  def get_next_regression : (Tensor,Tensor,Int) = {
    var inputs: List[List[Float]] = Nil
    var outputs: List[Float] = Nil
    var out = parseInput(inputIterartor, labelsIterartor, inputs, outputs, 1)
    while(batchSize > out._3.length) {out = parseInput(out._1, out._2, out._3, out._4, out._5)}
    ( Tensor(out._3).reshape( Shape(-1, inputDimensionExpected) ),
      Tensor(out._4).reshape( Shape(-1, outputDimensionExpected) ),
      out._5)
  }

  /**
    *
    * @return A batch for a classification problem
    */
  def get_next_classification : (Tensor,Tensor,Int) = {
    var inputs: List[List[Float]] = Nil
    var outputs: List[Float] = Nil
    var out = parseInput(inputIterartor, labelsIterartor, inputs, outputs, 1)
    while(batchSize > out._3.length) {out = parseInput(out._1, out._2, out._3, out._4, out._5)}
    val reshapedToClasses: List[List[Float]] = out._4.map {
      elem =>
        var auxL : Array[Float] = Array.fill[Float](outputDimensionExpected)(0.0.toFloat)
        auxL(elem.toInt) = 1.0.toFloat
        auxL.toList
    }
    ( Tensor(out._3).reshape(Shape(-1, inputDimensionExpected)),
      Tensor(reshapedToClasses).reshape(Shape(-1, outputDimensionExpected)),
      out._5)
  }
}

