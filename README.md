# inlamningsuppgift-python
# Del 2 – Dataanalys och strukturering av kod

Projektstruktur:

data/
    health_study_dataset.csv

del_2/
    utils.py
    health_analyzer.py
    health_analysis_del2.ipynb

src/
    README.md
    requirements.txt
   

Projektbeskrivning:

Strukturering av koden med funktioner och klassen HealthAnalyzer

Utökade visualiseringar: histogram, boxplot och stapeldiagram med en scatterplot

Enkel linjär regression och hypotestest (Welch's t-test)

Simulering av sjukdomsförekomst samt beräkning av konfidensintervall


Användning:

Projektet körs genom Jupyter-notebooken del2.ipynb i mappen del_2. Där läses datasetet in och ett HealthAnalyzer-objekt skapas som används för att utföra analyserna. Alla beräkningar och grafer är uppdelade i funktioner i utils.py och metoder i health_analyzer.py, som notebooken enkelt kan kalla på. Notebooken visar statistik, visualiseringar, t-test, konfidensintervall samt en linjär regression som förklarar blodtryck utifrån ålder och vikt.