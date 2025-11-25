import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def compute_basic_stats(df, columns):
    """
    Beräknar medelvärde, median, min och max för angivna kolumner i ett DataFrame.

    Args:
        df (DataFrame): Dataset med kolumner att analysera.
        columns (list): Lista med kolumnnamn.

    Returns:
        DataFrame: Statistik (medel, median, min, max) för kolumnerna.
    """
    return pd.DataFrame({
        'Medel': df[columns].mean(),
        'Median': df[columns].median(),
        'Min': df[columns].min(),
        'Max': df[columns].max()
}).round(2)


def plot_bp_hist(df):
    """
    Skapar ett histogram över systoliskt blodtryck.

    Args:
        df (DataFrame): Dataset med kolumnen 'systolic_bp'.
    """
    _, ax = plt.subplots(figsize=(8,5))  
    bp = df['systolic_bp'].dropna()       
    ax.hist(bp, bins=20, color='skyblue', edgecolor='black') 
    ax.set_title("Histogram av systoliskt blodtryck")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal deltagare")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)       
    plt.tight_layout()
    plt.show() 


def plot_weight_boxplot(df):
    """
    Skapar en boxplot över vikt uppdelad på kön.

    Args:
        df (DataFrame): Dataset med kolumnerna 'weight' och 'sex'.
    """
    _, ax = plt.subplots(figsize=(8,5))
    df_box = df[['weight','sex']].dropna()                    
    df_box.boxplot(column='weight', by='sex', ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', color='black'),
                   medianprops=dict(color='red'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'))
    ax.set_title("Boxplot över vikt per kön")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Vikt (kg)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.suptitle("")                                           
    plt.tight_layout()
    plt.show()


def plot_smokers_bar(df):
    """
    Skapar ett stapeldiagram över andelen rökare och icke-rökare.

    Args:
        df (DataFrame): Dataset med kolumnen 'smoker'.
    """
    _, ax = plt.subplots(figsize=(8,5))
    smoker_clean = df['smoker'].astype(str).str.strip().str.lower().map({'yes':'Yes','no':'No'})
    counts = smoker_clean.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='salmon', edgecolor='black', width=0.6)
    ax.set_title("Andel rökare")
    ax.set_xlabel("Rökare")
    ax.set_ylabel("Antal deltagare")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def scatter_age_bp(df):
    """
    Skapar en scatterplot av ålder mot systoliskt blodtryck.

    Args:
        df (DataFrame): Dataset med kolumnerna 'age' och 'systolic_bp'.
    """
    fig, ax = plt.subplots()
    ax.scatter(df['age'], df['systolic_bp'], alpha=0.6)
    ax.set_title("Ålder vs Systoliskt blodtryck")
    ax.set_xlabel("Ålder")
    ax.set_ylabel("Systoliskt blodtryck")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def simulate_disease(df, n=1000, seed=42):
    """
    Simulerar sjukdomsförekomst baserat på verklig andel i datasetet.

    Args:
        df (DataFrame): Dataset med kolumnen 'disease'.
        n (int): Antal personer som ska simuleras - 1000.
        seed (int): Seed för slump för reproducerbarhet.

    Returns:
        float: Simulerad andel personer med sjukdom.
    """
    p = df["disease"].mean()              
    np.random.seed(seed)
    simulated = np.random.binomial(1, p, n)
    return simulated.mean()


def bp_ci(df):
    """
    Beräknar 95% konfidensintervall för medelvärdet av systoliskt blodtryck.

    Args:
        df (DataFrame): Dataset med kolumnen 'systolic_bp'.

    Returns:
        tuple: Nedre och övre gräns för CI.
    """
    bp = df['systolic_bp'].dropna()
    mean_bp = bp.mean()
    se = bp.std(ddof=1) / len(bp)**0.5
    ci_lower = mean_bp - 1.96 * se
    ci_upper = mean_bp + 1.96 * se
    return ci_lower, ci_upper


def ttest_smokers(df):
    """
    Utför t-test för att jämföra blodtryck mellan rökare och icke-rökare.

    Args:
        df (DataFrame): Dataset med kolumnerna 'systolic_bp' och 'smoker'.

    Returns:
        tuple: t-statistik, p-värde, antal rökare, antal icke-rökare
    """
    df['smoker'] = df['smoker'].str.strip().str.lower()
    df_clean = df[['systolic_bp','smoker']].dropna()
    bp_smokers = df_clean[df_clean['smoker']=='yes']['systolic_bp']
    bp_nonsmokers = df_clean[df_clean['smoker']=='no']['systolic_bp']
    t_stat, p_value = stats.ttest_ind(bp_smokers, bp_nonsmokers, equal_var=False)
    return t_stat, p_value, len(bp_smokers), len(bp_nonsmokers)


def plot_bp_hist_smoker_groups(df):
    """
    Skapar ett dubbelhistogram av systoliskt blodtryck för rökare och icke-rökare.

    Args:
        df (DataFrame): Dataset med kolumnerna 'systolic_bp' och 'smoker'.
    """
    df = df.copy()
    df['smoker'] = df['smoker'].str.strip().str.lower()
    df_clean = df[['systolic_bp','smoker']].dropna()

    bp_smokers = df_clean[df_clean['smoker']=='yes']['systolic_bp']
    bp_nonsmokers = df_clean[df_clean['smoker']=='no']['systolic_bp']

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(bp_smokers, bins=25, alpha=0.6, label='Rökare', color='darkred')
    ax.hist(bp_nonsmokers, bins=25, alpha=0.4, label='Icke-rökare', color='red')

    ax.set_title("Histogram systoliskt blodtryck – rökare vs icke-rökare")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal deltagare")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()


def simple_regression(df):
    """
    Utför en enkel linjär regression för att visa blodtryck baserat på ålder och vikt.

    Args:
        df (DataFrame): Dataset med kolumnerna 'age', 'weight', 'systolic_bp'.

    Returns:
        LinearRegression: Tränad regressionsmodell.
    """
    df_clean = df.dropna(subset=['age','weight','systolic_bp'])
    x = df_clean[['age','weight']]
    y = df_clean['systolic_bp']
    model = LinearRegression()
    model.fit(x,y)
    return model


def plot_regression(df, model):
    """
    Skapar scatterplot av ålder vs blodtryck med regressionslinje.

    Prediktionen baseras på medelvärdet av vikt.

    Args:
        df (DataFrame): Dataset med kolumnerna 'age', 'weight', 'systolic_bp'.
        model (LinearRegression): Tränad regressionsmodell.
    """
    _, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df['age'], df['systolic_bp'], alpha=0.6, label="Data")

    df_clean = df.dropna(subset=['age','weight','systolic_bp'])
    ages = np.linspace(df_clean['age'].min(), df_clean['age'].max(), 100)
    weight_mean = df_clean['weight'].mean()
    X_pred = pd.DataFrame({'age': ages, 'weight': weight_mean})
    y_pred = model.predict(X_pred)

    ax.plot(ages, y_pred, color='red', label="Regression")
    ax.set_title("Ålder & Vikt vs Systoliskt blodtryck med regression")
    ax.set_xlabel("Ålder")
    ax.set_ylabel("Systoliskt blodtryck")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()


