## COVID-19 Statistics
<!--
![GitHub Stars](https://img.shields.io/github/stars/StevenHuang2020/WebSpider?label=Stars&style=social)
![GitHub watchers](https://img.shields.io/github/watchers/StevenHuang2020/WebSpider?label=Watch)
-->
![License: MIT](https://img.shields.io/badge/License-MIT-blue)
![Python Version](https://img.shields.io/badge/Python-v3-blue)
![Tensorflow Version](https://img.shields.io/badge/Tensorflow-V2.8-brightgreen)
![Last update](https://img.shields.io/endpoint?color=brightgreen&style=flat-square&url=https%3A%2F%2Fraw.githubusercontent.com%2FStevenHuang2020%2FWebSpider%2Fmaster%2Fcoronavirus%2Fupdate.json)

<!--footnotes define-->
[^1]: https://ourworldindata.org/covid-cases
[^2]: https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-case-demographics

<!--content start-->
### 📝 Contents
- [Usage](#Usage)
- [Statistics](#Statistics)
- [Reference](#Reference)

### Usage

pip install -r requirements.txt<br/>

|CMD|Description|
|---|---|
|python main.py|#visualize world covid-19 statistics|
|python main_NZ.py |#visualize New Zealand covid-19 statistics|
|python predict_cases.py|#predict world covid-19 cases |


### Vaccination Stats

|||
|---|---|
|<img src="images/World_vaccinatedPerHundred.png" width="320" height="240" />|<img src="images/World_vaccinated.png" width="320" height="240" />|
|<img src="images/World_vaccinatedNew.png" width="320" height="240" />|<img src="images/World_vaccinatedTotal.png" width="320" height="240" />|
|<img src="images/World_vaccineRankingPeople.png" width="320" height="240" />|<img src="images/World_vaccineRankingPeoplePerH.png" width="320" height="240" />|
|<img src="images/World_vaccineFully_top.png" width="320" height="240" />|<img src="images/World_vaccinePerH_top.png" width="320" height="240" />|
|<img src="images/World_peopleVaccined_top.png" width="320" height="240" />|<img src="images/World_vaccineContinent.png" width="320" height="240" />|
|<img src="images/World_peopleVaccined_topCasesCountries.png" width="320" height="240" />|<img src="images/World_peopleVaccinedPerH_topCasesCountries.png" width="320" height="240" />|
|<img src="images/continentTopCountries_vaccinePH.png" width="320" height="240" />|<img src="images/continentTopCountries_vaccineFully.png" width="320" height="240" />|


### Case Stats

|||
|---|---|
|<img src="images/1.png" width="320" height="240" />|<img src="images/2.png" width="320" height="240" />|
|<img src="images/3.png" width="320" height="240" />|<img src="images/4.png" width="320" height="240" />|
|<img src="images/5.png" width="320" height="240" />|<img src="images/6.png" width="320" height="240" />|
|<img src="images/7.png" width="320" height="240" />||
|<img src="images/World_casesContinent.png" width="320" height="240" />|<img src="images/World_newCasesContinent.png" width="320" height="240" />|
|<img src="images/continentTopCountries_NewCases.png" width="320" height="240" />|<img src="images/continentTopCountries_NewDeaths.png" width="320" height="240" />|

<br/>

|||
|---|---|
|<img src="images/countries_Confirmed.png" width="320" height="240" />|<img src="images/countries_NewConfirmed.png" width="320" height="240" />|
|<img src="images/countries_Deaths.png" width="320" height="240" />|<img src="images/countries_NewDeaths.png" width="320" height="240" />|
|<img src="images/World_Cases.png" width="320" height="240" />|<img src="images/World_NewCases.png" width="320" height="240" />|
|<img src="images/World_RecentNewCases.png" width="320" height="240" />|<img src="images/World_Deaths.png" width="320" height="240" />|
|<img src="images/World_NewDeaths.png" width="320" height="240" />|<img src="images/World_RecentNewDeaths.png" width="320" height="240" />|
|<img src="images/continent_NewConfirmed.png" width="320" height="240" />|<img src="images/continent_NewDeaths.png" width="320" height="240" />|
|<img src="images/World_Mortality.png" width="320" height="240" />|<img src="images/countries_MortalityTC.png" width="320" height="240" />|

<br/>

### World cases Prediction
Predicting cases using the LSTM algorithm.<br/>
Data Source[^1], dataset [here.](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv)<br/>
<br/>

|||
|---|---|
|<img src="images/WorldPredictCompare.png" width="320" height="240" />|<img src="images/WorldFuturePredict.png" width="320" height="240" />|
|<img src="images/WorldFuturePredictPrecise.png" width="320" height="240" />||


### NZ Covid-19 Statistic
Data source reference [here.](https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-case-demographics)

|||
|---|---|
|<img src="images/NZ_Gender.png" width="320" height="240" />|<img src="images/NZ_DHB.png" width="320" height="240" />|
|<img src="images/NZ_AgeGroup.png" width="320" height="240" />|<img src="images/NZ_COVID-19_RecentCases.png" width="320" height="240" />|
|<img src="images/NZ_COVID-19_EveryDayCases.png" width="320" height="240" />|<img src="images/NZ_COVID-19_CumlativeCases.png" width="320" height="240" />|
|<img src="images/NZ_IsOVerseas.png" width="320" height="240" />||


## References

 - Our World in Data[^1], datasheet dowload [here.](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv) <br/>
 - NZ Ministry of Health[^2], datasheet refer to [here.](https://www.health.govt.nz/our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-case-demographics )
