# <center>Spark project MS Big Data Télécom : Kickstarter campaigns</center>

*Thomas Koch*

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020

## Introduction

Deux notebook de travail sont disponibles pour faciliter le lancement des codes `Preprocessor.scala` et `Trainer.scala`. Ils sont exécutables dans Jupyter à condition d'avoir installé au préalable un [spylon-kernel](https://pypi.org/project/spylon-kernel/0.1.1/). Ils permettront, si besoin, de retravailler le code, de faire facilement des tests et petites requêtes, avant un déploiement à plus grande échelle. 

Le premier notebook correspond au code **Preprocessor** et le second au code **Trainer**.

Les parties qui suivent visent à mettre l'accent sur les différences de méthodes utilisées par rapport aux instructions données dans les [énoncés de TP](https://github.com/Flooorent/cours-spark-telecom).

## 1. Remarques sur le nettoyage des données

### 1.1 Colonne *final_status*
 
En lançant la requête suivante :

* 	```scala
	dfCountry.groupBy($"final_status" > 1).count.orderBy($"count".desc).show()
	``` 
J'ai pu constater que les final_status, n'étaient pas tous à 1 (Success) ou 0 (Fail).
* 	```scala
	+-------------------+------+
	|(final_status > 1)| count|
	+-------------------+------+
	|              false|107685|
	|               true|   443|
	|               null|     1|
	+-------------------+------+
	```

L'énoncé du TP nous invitait à conserver uniquement les lignes qui nous intéressent pour le modèle, à savoir lorsque ***final_status*** vaut 0 (Fail) ou 1 (Success).

Toutefois, j'ai préféré adopter une autre approche, en considérant que les campagnes qui ne sont pas un Success sont un Fail. Aussi, j'ai choisi de passer à 0 toutes les valeurs plus grandes que 1. Le code est :

*	```scala
	def cleanFinalStatus(final_status: Int): Int = {
     	if (final_status != 1)
       	  0
     	else
       	  final_status
	}

	val cleanFinalStatusUdf = udf(cleanFinalStatus _)

	val dfCountry2: DataFrame = dfCountry
  	    	.withColumn("final_status", cleanFinalStatusUdf($"final_status"))
  	    	.filter($"final_status" === 0 || $"final_status" === 1)
	```

Le `.filter` final me permet de filtrer la valeur ***null*** que je n'ai pas traité dans ma fonction.

A la fin du `Preprocessor`, j'obtiens :

*	```scala
	dfCountry2.groupBy($"final_status" > 1).count.orderBy($"count".desc).show()
	
	+------------------+------+
	|(final_status > 1)| count|
	+------------------+------+
	|             false|108128|
	+------------------+------+
	```
	```scala
	dfCountry2.groupBy("final_status").count.orderBy($"count".desc).show()
	
	+------------+-----+
	|final_status|count|
	+------------+-----+
	|           0|73709|
	|           1|34419|
	+------------+-----+
	```
Ce résultats me prouvent donc bien la validité de ma fonction.


### 1.2 Colonne *hours_prepa*

En ajoutant la colonne ***hours_prepa*** qui représente le nombre d’heures de préparation de la campagne entre ***created_at*** et ***launched_at***, je me suis rendu compte qu'il y avait des résultats négatifs, dus à une **date de lancement antérieure à la date de création**. 

Deux possibilités s'offraient à moi pour traiter cette incohérence : 
* Supprimer les lignes des projets ayant le problème ;
* Fixer la date de création égale à celle de lancement ;

Même si la requête 
*	```scala
	dfCountry3.groupBy($"hours_prepa" < 0).count.orderBy($"count".desc).show()
	```
me renvoyait un faible nombre de projet dans ce cas :
* 	```scala 
	+-----------------+------+
	|(hours_prepa < 0)| count|
	+-----------------+------+
	|            false|107615|
	|             true|    70|
	+-----------------+------+
	```
j'ai préféré opter pour la deuxième option, pour me familiariser un peu plus avec les `udf`. Aussi, le code que j'ai rédigé est :
*	```scala
	def cleanHoursPrepa(created_at: Int, launched_at: Int, hours_prepa: Int): Int = {
	  if (hours_prepa < 0)
	    launched_at
	  else
	    created_at
	}

	val cleanHoursPrepaUdf = udf(cleanHoursPrepa _)

	val dfCountry2: DataFrame = dfCountry
	  .withColumn("created_at2", cleanHoursPrepaUdf($"created_at", $"launched_at", $"hours_prepa"))
	  .withColumn("hours_prepa", ((($"launched_at" - $"created_at2")/3.6).cast("Int")/1000))
	  .drop("created_at")
	```

Et en lançant la requête 
*	```scala
	dfCountry2.groupBy($"hours_prepa" < 0).count.orderBy($"count".desc).show()
	```
J'ai eu le résultat :
*	```scala
	+-----------------+------+
	|(hours_prepa < 0)| count|
	+-----------------+------+
	|            false|107685|
	+-----------------+------+
	```
Me prouvant donc bien le bon fonctionnement de ma fonction.


## 2. Analyse des résultats obtenus avec ma méthode de nettoyage

Après avoir rédigé le `Trainer.scala`, j'ai lancé une première fois le modèle avec le jeu de données fourni, et une deuxième fois avec mon jeu de données perso. J'ai ainsi pu constater une légère amélioration des résultats.

* **Données fournies** : après entraînement du modèle et réglage des hyper-paramètres, on obtient
	```
	+------------+-----------+-----+
	|final_status|predictions|count|
	+------------+-----------+-----+
	|           1|        0.0| 1049|
	|           0|        1.0| 2899|
	|           1|        1.0| 2361|
	|           0|        0.0| 4493|
	+------------+-----------+-----+

	Le f1 score est de 0.647367360180229
	```
* **Données perso** : après entraînement du modèle et réglage des hyper-paramètres, on obtient 
	```scala
	+------------+-----------+-----+
	|final_status|predictions|count|
	+------------+-----------+-----+
	|           1|        0.0| 1003|
	|           0|        1.0| 2740|
	|           1|        1.0| 2427|
	|           0|        0.0| 4623|
	+------------+-----------+-----+

	Le f1 score est de 0.6650475698419106 
	```
On a donc globalement **augmenté le nombres de bonnes prédictions**, et **diminué le nombre de mauvaises**, ce qui s'en ressent avec **quelques points supplémentaires dans le f1-score**.

## Conclusion

Ce travail m'a permis de voir l'importance du nettoyage et de la préparation des données avant de pouvoir les utiliser correctement dans un modèle de ML. 

Quant au travail de Machine Learning, il demande du temps de calcul et donc de la patience pour réussir à trouver le modèle et ses hyper-paramètres les plus adaptés à nos données.

L'ensemble de ces TP et cours était très enrichissant pour une première expérience avec Spark, très bien documenté et très bien cadré. 

Bravo à toute l'équipe pour le travail fourni dans la préparation et la réalisation, c'est vraiment pro et ça méritait d'être souligné !

