package com.typeform.utils

import java.io._
import javax.swing.JPopupMenu.Separator
import org.platanios.tensorflow.api.{FLOAT32, _}
import scala.util.Random
object Utils {

  def createDatasetForClassification(xDim:Int,
                                     yNumClasses:Int,
                                     filLocationX:String,
                                     filLocationY:String,
                                     separator: String,
                                     records:Int):Unit= {

    val random = new Random()
    val trueParams : Array[Double] = Array.fill[Double](xDim)(random.nextFloat())
    val encoding: String = "UTF-8"
    val maxlines: Int = 100
    new File(filLocationX).delete()
    var writerX: BufferedWriter = null
    new File(filLocationY).delete()
    var writerY: BufferedWriter = null
    writerX = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filLocationX), encoding))
    writerY = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filLocationY), encoding))
    for( i <- 0 to records) {
      val input : Array[Double] = Array.fill[Double](xDim)(random.nextFloat())
      val res = input.zip(trueParams).map(elem => elem._1 * elem._2 )
      writerX.write(input.mkString(separator))
      writerX.newLine()
      writerY.write(scala.math.floor((res.sum / xDim.toDouble)*yNumClasses.toDouble).toString)
      writerY.newLine()

    }
    writerY.close()
    writerX.close()
  }

  def batchNormalization(O:Output) : Output = {
    val moments = tf.moments(O,Seq[Int](0))
    val normalized:Output  = tf.divide(tf.ones(FLOAT32,moments._2.shape)*(O - moments._1),moments._2 + tf.ones(FLOAT32,moments._2.shape)) + tf.zeros(FLOAT32,moments._2.shape)
    normalized
  }

}
