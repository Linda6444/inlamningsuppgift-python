import utils

class HealthAnalyzer:   
    """
    Klass för att analysera hälsodata. 
    Kan beräkna statistik, skapa grafer, simulera sjukdom, 
    utföra t-test och regressionsanalyser.
    """
    def __init__(self, df):
        """
        Initierar HealthAnalyzer med ett dataset.

        Args:
            df (DataFrame): Dataset med hälsodata.
        """
        self.df = df
     

    def stats(self, columns):
        return utils.compute_basic_stats(self.df, columns)
    """
        Beräknar grundläggande statistik för valda kolumner.

        Args:
            columns (list): Lista med kolumnnamn.

        Returns:
            DataFrame: Statistik (medel, median, min, max).
    """


    def histogram_bp(self):
        """Ritar histogram över systoliskt blodtryck."""
        utils.plot_bp_hist(self.df)
        

    def boxplot_weight(self):
        """Ritar boxplot över vikt per kön."""
        utils.plot_weight_boxplot(self.df)

    def bar_smokers(self):
        """Ritar stapeldiagram över rökare."""
        utils.plot_smokers_bar(self.df)

    def scatter_age_bp(self):
        """Ritar scatterplot över ålder vs systoliskt blodtryck."""
        utils.scatter_age_bp(self.df)

  
    def simulate_disease(self, n=1000, seed=42):
        """
        Simulerar sjukdomsförekomst.

        Args:
            n (int): Antal personer som ska simuleras - 1000.
            seed (int): Seed för slumpgenerator.

        Returns:
            float: Simulerad andel sjukdom.
        """
        return utils.simulate_disease(self.df, n=n, seed=seed)
    
 
    def ci_bp(self):
        """
        Beräknar 95% konfidensintervall för medelvärdet av systoliskt blodtryck.

        Returns:
            tuple: Nedre och övre gräns för CI.
        """
        ci_low, ci_high = utils.bp_ci(self.df) 
        print(f"95% CI för systoliskt blodtryck: [{ci_low:.2f}, {ci_high:.2f}]")
        return ci_low, ci_high

   
    def ttest_smokers(self):
        """
        Utför t-test för att jämföra blodtryck mellan rökare och icke-rökare.

        Returns:
            tuple: t-statistik, p-värde, antal rökare, antal icke-rökare
        """
        return utils.ttest_smokers(self.df)
    
    
    def histogram_smoker_groups(self):
        """Ritar dubbelhistogram över systoliskt blodtryck för rökare och icke-rökare."""
        utils.plot_bp_hist_smoker_groups(self.df)


    def regression_age_weight_bp(self):
        """
        Utför enkel linjär regression för blodtryck baserat på ålder och vikt.

        Returns:
            LinearRegression: Tränad regressionsmodell.
        """
        return utils.simple_regression(self.df)

 
    def plot_regression(self):
        """
        Skapar scatterplot över ålder vs blodtryck med regressionslinje.
        Prediktionen baseras på medelvikt.
        """
        model = self.regression_age_weight_bp()
        utils.plot_regression(self.df, model)

        



    


    
