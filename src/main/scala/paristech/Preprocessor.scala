package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      * import spark.implicits._
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
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
      .csv("src/main/ressources/data/train_clean.csv")

    println("\ndf : ")
    df.printSchema
    println("df : ")
    df.show(5)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Assignation des types Int aux colonnes adequates             //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))
    
    println("\ndfCasted : ")
    dfCasted.printSchema
    println("dfCasted : ")
    dfCasted.show(5)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Cleaning fuites du futur et disable_communication            //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    val dfNoFutur: DataFrame = dfCasted
      .drop("disable_communication")
      .drop("backers_count", "state_changed_at")

    println("\ndfNoFutur : ")
    dfNoFutur.printSchema
    println("dfNoFutur : ")
    dfNoFutur.show(5)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Cleaning colonnes currency, country et final_status          //")
    println("//                    Creation des colonnes days_campaign et hours_prepa           //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

   def cleanFinalStatus(final_status: Int): Int = {
     if (final_status != 1)
       0
     else
       final_status
}

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)
    val cleanFinalStatusUdf = udf(cleanFinalStatus _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .withColumn("final_status", cleanFinalStatusUdf($"final_status"))
      .drop("country", "currency")
      .filter($"final_status" === 0 || $"final_status" === 1)
      .withColumn("days_campaign", datediff(from_unixtime($"deadline") , from_unixtime($"launched_at")))
      .withColumn("hours_prepa", (($"launched_at" - $"created_at")/3.6).cast("Int")/1000)

    println("\ndfCountry : ")
    dfCountry.printSchema
    println("dfCountry : ")
    dfCountry.show(5)


    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Cleaning colonnes name, desc, keywords, hours prepa,         //")
    println("//                      created_at, deadline et launched_at                        //")
    println("//                    Cleaning des valeurs nulles                                  //")
    println("/////////////////////////////////////////////////////////////////////////////////////")

    def cleanHoursPrepa(created_at: Int, launched_at: Int, hours_prepa: Int): Int = {
      if (hours_prepa < 0)
        launched_at
      else
        created_at
    }

    val cleanHoursPrepaUdf = udf(cleanHoursPrepa _)

    val dfCountry2: DataFrame = dfCountry
      .withColumn("created_at2", cleanHoursPrepaUdf($"created_at", $"launched_at", $"hours_prepa"))
      .withColumn("hours_prepa", (($"launched_at" - $"created_at2")/3.6).cast("Int")/1000)
      .drop("created_at")
      .drop("launched_at", "created_at2", "deadline")
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))
      .na.fill(Map("currency2" -> "unknown", "country2" -> "unknown", "days_campaign" -> -1, "hours_prepa" -> -1, "goal" -> -1))

    println("\ndfCountry2 : ")
    dfCountry2.printSchema
    println("dfCountry2 : ")
    dfCountry2.show(5)

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    Sauvegarde du DataFrame => monDataFrameFinal                 //")
    println("/////////////////////////////////////////////////////////////////////////////////////")


    val monDataFrameFinal: DataFrame = dfCountry2

    monDataFrameFinal.write.mode("overwrite").parquet("src/main/ressources/monDataFrameFinal")

  }
}
