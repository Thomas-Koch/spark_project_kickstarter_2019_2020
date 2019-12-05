package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, OneHotEncoderEstimator, RegexTokenizer, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StopWordsRemover

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}

import java.io.File

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    TP 3 : Machine learning avec Spark                           //")
    println("/////////////////////////////////////////////////////////////////////////////////////")


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Chargement des donnees                                       //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    val df: DataFrame = spark
      .read
      .option("header", value = true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet("src/main/ressources/monDataFrameFinal") //data/prepared_trainingset")  monDataFrameFinal

    println("Training Dataframe")
    println(df.show(5))
    println(df.printSchema)

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Utilisation des donnees textuelles                           //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    println("//                    Etape 1 : Separation des textes en mots                      //")
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    println("//                    Etape 2 : Retirage des stop words                            //")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    println("//                    Etape 3 : Conversion en TF-IDF                               //")
    val countVectorizedModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vectorized")

    println("//                    Etape 4 : Recherche de la partie IDF                         //")
    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Conversion des variables catégorielles en variables numeriques/")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    println("//                    Etape 1 : Conversion de country2                             //")
    val stringIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    println("//                    Etape 2 : Conversion de currency2                            //")
    val stringIndexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    println("//                    Etape 3 : One-Hot encoder de ces deux catégories             //")
    val oneHotEncoder = new OneHotEncoderEstimator() // OneHotEncoder étant deprecated, on utilise OneHotEncoderEstimator
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Mise des donnees sous un forme interpretable par SparkML     //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    println("//                    Etape 1 : VectorAssembler                                    //")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    println("OUTPUT FEATURES")

    println("//                    Etape 2 : Creation et instanciation du modèle de classification")
    println("//                    Regression logistique                                        //")
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Création du Pipeline                                         //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    val stages10 = Array(tokenizer, stopWordsRemover, countVectorizedModel, idf, stringIndexer, stringIndexer2, oneHotEncoder, vectorAssembler, lr)
    val pipeline = new Pipeline().setStages(stages10)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Entrainement, test et evaluation du modele                   //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 1991)

    val model = pipeline.fit(training)
    println(s"Model 1 was fit using parameters: ${model.parent.extractParamMap}")

    val dfWithSimplePredictions = model.transform(test)

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val f1score = evaluator.evaluate(dfWithSimplePredictions)

    println("Le f1-score est de " + f1score)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(countVectorizedModel.minDF, Array(55.0, 75.0, 95.0))
      .build()

    //  TrainValidationSplit requiert un estimateur, un set d'estimateur ParamMaps, et un Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Entrainement du modèle avec l'échantillon training
    println("Entrainement du modèle avec l'échantillon training")
    val validationModel = trainValidationSplit.fit(training)

    val dfWithPredictions = validationModel.transform(test).select("features","final_status","predictions")

    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    val score = evaluator.evaluate(dfWithPredictions)

    dfWithPredictions.groupBy("final_status","predictions").count.show()

    println("F1 Score est " + score)


    // Evaluer la precision (accuracy)
    val evaluator_acc = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

    // Obtention de la mesure de performance
    val accuracy = evaluator_acc.evaluate(dfWithPredictions)
    println("Precision obtenue : " + accuracy)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Sauvegarde du modele                                         //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    // Saving model

    validationModel.write.overwrite.save("src/main/model/LogisticRegression")


  }
}
