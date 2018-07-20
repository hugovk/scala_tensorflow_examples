
import java.io.File
import java.nio.file.Paths

import com.tyepform.iterators._
import com.typesafe.scalalogging.Logger
import org.platanios.tensorflow.api.{FLOAT32, _}
import org.slf4j.LoggerFactory
import com.typeform.utils.Utils.batchNormalization
import scala.util.Random

object SequentialDNNForClassification extends App {

  val logger = Logger(LoggerFactory.getLogger(getClass().getName))
  logger.info("Parameters for the model.")
  //Defining parameters for network and the DataIterator
  val params : Map[String,String] = Map[String,String]( "filesFolder" -> "/Users/felipemateos/work/DataForTesting/regressionInTensorFlow",
    "inputsFile"-> "dnnInputs.csv",
    "labelsFile" -> "dnnLabels.csv",
    //"separator" -> "\\t",
    "separator" -> ",",
    "batchSize" -> "128",
    "inputDimensionExpected" -> "500",
    "hiddenLayersDimensionExpected" -> "400,300,50",
    "outputDimensionExpected" -> "10",
    "cut" -> "0.5" ,
    "dropOutProb" -> "0.8"
  )
  //Creating  testFile
  com.typeform.utils.Utils.createDatasetForClassification(
    params("inputDimensionExpected").toInt,
    params("outputDimensionExpected").toInt,
    s"${params("filesFolder")}${File.separator}${params("inputsFile")}",
    s"${params("filesFolder")}${File.separator}${params("labelsFile")}",
    params("separator"),
    10000
  )

  params.keySet.foreach( elem => logger.info(s"$elem => ${params(elem)}") )
  // Creating a Object to read the data from file
  val bi : BasicIteratorFromFileInputString = new BasicIteratorFromFileInputString(params)
  //Setting the number of times i would like to go through the dataset.
  val epochs = 80
  //Creating placeholders for the Input and the labels
  val input = tf.placeholder(FLOAT32, Shape(-1, params("inputDimensionExpected").toInt))
  val output = tf.placeholder(FLOAT32, Shape(-1, params("outputDimensionExpected").toInt))
  //Setting dictionaries where the weights , bias  ( Network parameters to be learned )
  var weightsMap : Map[String,Variable] = Map[String,Variable]()
  var biasMap : Map[String,Variable] = Map[String,Variable]()
  //Setting activation values for all latyers.
  var inputsMap :  Map[String,Output] = Map[String,Output]("0_input" -> input)
  //dropoutLayers:
  var inputsWithDropOutsMap :  Map[String,Output] = Map[String,Output]()
  //Normailized input
  var normalizedInputMap :  Map[String,Output] = Map[String,Output]("0_normalizedInput" -> batchNormalization(input))
  //Formating information regarding the Network Arquitectute . HiddenLayers , and sizes
  val dimensions : List[Int] =  (
    params("outputDimensionExpected").toInt  ::
    ( params("inputDimensionExpected").toInt
      :: params("hiddenLayersDimensionExpected").split(",").map(_.toInt).toList ).
      reverse ).
    reverse
  //Seting the size of the matrix for the paremeters to be learned
  var sizesTensorParameters : List[((Int,Int),Int)] = dimensions.zip(dimensions.tail).zipWithIndex
  //
  for( ((iSize,oSize),lIndex) <- sizesTensorParameters){

    weightsMap += (s"${lIndex}_weights" -> tf.variable(s"weightsInputToHidden$lIndex",FLOAT32, Shape(iSize,oSize),  tf.RandomTruncatedNormalInitializer()))
    biasMap    += (s"${lIndex}_bias" -> tf.variable(s"biasInputToHidden$lIndex", FLOAT32, Shape(oSize),  tf.RandomTruncatedNormalInitializer() ) )

  }
  for( i <-  1 to sizesTensorParameters.length  ){
     if (i == sizesTensorParameters.length ){
       //inputsMap += ( s"${i}_input" -> ( tf.matmul( inputsMap(s"${i - 1}_input"), weightsMap(s"${i - 1}_weights") ) + biasMap(s"${i - 1}_bias") ) )
       inputsMap += ( s"${i}_input" -> ( tf.matmul( inputsWithDropOutsMap(s"${i - 1}_dropout"), weightsMap(s"${i - 1}_weights") ) + biasMap(s"${i - 1}_bias") ) )
     }
     else if(i == 1){
       inputsMap += ( s"${i}_input" -> tf.relu( tf.matmul( inputsMap(s"${i - 1}_input"), weightsMap(s"${i - 1}_weights") )  + biasMap(s"${i - 1}_bias" ) ) )
       normalizedInputMap +=  ( s"${i}_normalizedInput" -> batchNormalization(inputsMap(s"${i}_input") ) )
       //inputsWithDropOutsMap += (s"${i}_dropout" -> tf.dropout(inputsMap(s"${i}_input"),params("dropOutProb").toFloat))
       inputsWithDropOutsMap += (s"${i}_dropout" -> tf.dropout(normalizedInputMap(s"${i}_normalizedInput"),params("dropOutProb").toFloat))
     }
     else {
       // inputsMap += ( s"${i}_input" -> tf.relu( tf.matmul( inputsMap(s"${i - 1}_input"), weightsMap(s"${i - 1}_weights") )  + biasMap(s"${i - 1}_bias" ) ) )
       inputsMap += ( s"${i}_input" -> tf.relu( tf.matmul( inputsWithDropOutsMap(s"${i - 1}_dropout"), weightsMap(s"${i - 1}_weights") )  + biasMap(s"${i - 1}_bias" ) ) )
       normalizedInputMap +=  ( s"${i}_normalizedInput" -> batchNormalization(inputsMap(s"${i}_input") ) )
       //inputsWithDropOutsMap += (s"${i}_dropout" -> tf.dropout(inputsMap(s"${i}_input"),params("dropOutProb").toFloat))
       inputsWithDropOutsMap += (s"${i}_dropout" -> tf.dropout(normalizedInputMap(s"${i}_normalizedInput"),params("dropOutProb").toFloat))

     }
  }

  val predictions : Output = inputsMap(s"${sizesTensorParameters.length}_input")
  val loss :(Output,Output,Output) = OwnLost(predictions , output)
  val trainOp  = tf.train.
    AdaGrad(1.0).
    minimize(loss._1)


  //TODO MODIFY THE FUNCTION , EXPONENTIAL BAD

  def OwnLost(predicted:Output, actual :Output) : (Output,Output,Output) ={
    //val clippedPrediction : Output = tf.clipByValue(predictions,-15.0.toFloat,15.0.toFloat)
    //tf.sum(predicted)
    //to avoid numerical overflow
    val max = tf.max(predicted,1)
    val res= predicted - tf.reshape(max,Shape(-1,1))
    val denom : Output = tf.sum(tf.exp(res),1)
    val m : Output = tf.exp(res) / tf.reshape(denom,Shape(-1,1))
    (tf.sum(tf.square(m - actual)),max,m)
  }

  val session = Session()
  session.run(targets = tf.globalVariablesInitializer())
  var linesReadInTotal = 0
  var printedTheEpoch = true
  val realEpochs = ( epochs / params("cut").toDouble ).toInt
  for (i <- 0 to realEpochs ) {
    printedTheEpoch = true
    while (i == bi.epoch) {
      val trainBatch = bi.get_next_classification
      linesReadInTotal +=  trainBatch._3
      val feeds: Map[Output, Tensor] = Map(input -> trainBatch._1, output -> trainBatch._2)
      val trainLoss: (Tensor,Tensor,Tensor) = session.run(feeds = feeds, fetches = loss , targets = trainOp)
      if (printedTheEpoch)
        logger.info(s"Train loss at epoch ${ i * params("cut").toDouble } = ${trainLoss._1.summarize(flattened = true)} " + s"( weight last Layer = ${session.run(fetches = weightsMap(s"${sizesTensorParameters.length - 1 }_weights").value)} )" +
          s"With $linesReadInTotal records read in total")
      printedTheEpoch=false
    }
  }
  logger.info(s"Trained weight value: ${session.run(fetches = weightsMap(s"${sizesTensorParameters.length - 1}_weights").value).summarize(flattened = true)}")
  logger.info(s"Trained bias   value: ${session.run(fetches = biasMap(s"${sizesTensorParameters.length - 1 }_weights").value).summarize(flattened = true)}")

}
