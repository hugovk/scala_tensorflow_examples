
import java.nio.file.Paths

import org.platanios.tensorflow.api.{FLOAT32, _}
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import scala.util.Random
import com.tyepform.iterators._

object SimpleLinialRegression extends App {

  val params : Map[String,String] = Map[String,String]( "filesFolder" -> "/Users/felipemateos/work/DataForTesting/regressionInTensorFlow",
    "inputsFile"-> "inputs.tab",
    "labelsFile" -> "labels.tab",
    "separator" -> "\\t",
    "batchSize" -> "128",
    "inputDimensionExpected" -> "3",
    "outputDimensionExpected" -> "1",
    "cut" -> "0.5"
  )
  val epochs = 80
  val random = new Random()
  val logger = Logger(LoggerFactory.getLogger("Examples / Linear Regression"))
  logger.info("Building linear regression model.")
  val inputs = tf.placeholder(FLOAT32, Shape(-1, 3))
  val outputs = tf.placeholder(FLOAT32, Shape(-1, 1))
  val weights = tf.variable("weights", FLOAT32, Shape(3,1), tf.ZerosInitializer)
  val bias = tf.variable("bias", FLOAT32, Shape(1,1), tf.ZerosInitializer)
  val predictions :Output = tf.matmul(inputs, weights) + bias
  val loss = OwnLost(predictions , outputs)

  val lossStats = tf.learn.ScalarSummary("loss","loss")
  val ssummaryFile = Paths.get("/tmp/summaries")

  val trainOp = tf.train.
    AdaGrad(1.0).
    minimize(loss)

  /**
    * A way of defining your own loss functions
    * @param predicted Tensor with the predicted labels
    * @param actual actual labels
    * @return
    */
  def OwnLost(predicted:Output, actual :Output) : Output ={
    tf.sum(tf.square(predicted - outputs))
  }

  logger.info("Training the linear regression model.")

  val session = Session()
  val bi : BasicIteratorFromFileInputString = new BasicIteratorFromFileInputString(params)
  session.run(targets = tf.globalVariablesInitializer())
  var linesReadInTotal = 0
  var printedTheEpoch = true
  val realEpochs = ( epochs / params("cut").toDouble ).toInt
  for (i <- 0 to realEpochs ) {
    printedTheEpoch = true
    while (i == bi.epoch) {
      val trainBatch = bi.get_next_regression
      linesReadInTotal +=  trainBatch._3
      val feeds: Map[Output, Tensor] = Map(inputs -> trainBatch._1, outputs -> trainBatch._2)
      val trainLoss: Tensor = session.run(feeds = feeds, fetches = loss, targets = trainOp)
      if (printedTheEpoch)
        logger.info(s"Train loss at epoch ${ i * params("cut").toDouble } = ${trainLoss.scalar} " + s"(weight = ${session.run(fetches = weights.value)} )" +
          s"With ${linesReadInTotal} records read in total")
        printedTheEpoch=false
    }
  }
  logger.info(s"Trained weight value: ${session.run(fetches = weights.value).summarize(flattened = true)}")
  logger.info(s"Trained bias value: ${session.run(fetches = bias.value).summarize(flattened = true)}")

  /*
  def batch(batchSize:Int) : (Tensor,Tensor) ={

    var inputs : List[Float] = Nil
    var outputs : List[Float] = Nil
    (0 to batchSize).foreach{
      elem =>
        val r : Float= random.nextFloat()
        inputs = r :: inputs
        outputs = r * weight :: outputs
    }
    (Tensor(inputs).reshape(Shape(-1,1)),Tensor(outputs).reshape(Shape(-1,1)))
  }
*/
}
