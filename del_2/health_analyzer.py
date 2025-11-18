import utils
import pandas as pd


# Klass för att analysera hälsodata.
class HealthAnalyzer:   

    def __init__(self, df: pd.DataFrame): 
        self.df = df # sparar DataFrame som attribut

 # Statistik - Beräknar medel, median, min och max för valda kolumner"
    def stats(self, columns):
        return utils.compute_basic_stats(self.df.columns)

  # Grafer
    def histogram_bp(self):
        utils.plot_bp_hist(self.df)

    def boxplot_weight(self):
        utils.plot_weight_boxplot(self.df)

    def bar_smokers(self):
        utils.plot_smokers_bar(self.df)

    # Simulering
    def simulate_disease(self, n=1000, seed=42):
        return utils.simulate_disease_probability(self.df, n=n, seed=seed)
    
    # Hypotesprövning
    def ttest_smokers(self):
        return utils.ttest_smokers(self.df)
    
    # Enkel regression
    def regression_age_weight_bp(self):
        return utils.simple_regression(self.df)

    


    
