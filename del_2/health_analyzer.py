import utils


# Klass för att analysera hälsodata.
class HealthAnalyzer:   

   
    def __init__(self, df):
        self.df = df

    def stats(self, columns):
        return utils.compute_basic_stats(self.df, columns)


  # Grafer
    def histogram_bp(self):
        utils.plot_bp_hist(self.df)

    def boxplot_weight(self):
        utils.plot_weight_boxplot(self.df)

    def bar_smokers(self):
        utils.plot_smokers_bar(self.df)

    def scatter_age_bp(self):
        utils.scatter_age_bp(self.df)

  # Simulering
    def simulate_disease(self, n=1000, seed=42):
        return utils.simulate_disease(self.df, n=n, seed=seed)
    
  # Konfidensintervall 
    def ci_bp(self):
        ci_low, ci_high = utils.bp_ci(self.df) 
        print(f"95% CI för systoliskt blodtryck: [{ci_low:.2f}, {ci_high:.2f}]")
        return ci_low, ci_high

    # Hypotestest / t-test
    def ttest_smokers(self):
        return utils.ttest_smokers(self.df)

    # Regression
    def regression_age_weight_bp(self):
        return utils.simple_regression(self.df)

    


    
