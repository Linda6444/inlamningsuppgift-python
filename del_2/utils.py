import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

#Beräknar medel, median, min och max för valda kolumner.
def compute_basic_stats(df, columns):
    return pd.DataFrame({
    'Medel': df[columns].mean(),
    'Median': df[columns].median(),
    'Min': df[columns].min(),
    'Max': df[columns].max()
}).round(2)

# Histogram över systoliskt blodtryck
def plot_bp_hist(df):
    _, ax = plt.subplots(figsize=(8,5))  
    bp = df['systolic_bp'].dropna()       
    ax.hist(bp, bins=20, color='skyblue', edgecolor='black') 
    ax.set_title("Histogram av systoliskt blodtryck")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal deltagare")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)       
    plt.tight_layout()
    plt.show() 

# Boxplot för vikt per kön
def plot_weight_boxplot(df):
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

# Stapeldiagram för rökare
def plot_smokers_bar(df):
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

# Scatterpot för ålder vs blodtrtyck
def scatter_age_bp(df):
    fig, ax = plt.subplots()
    ax.scatter(df['age'], df['systolic_bp'], alpha=0.6)
    ax.set_title("Ålder vs Systoliskt blodtryck")
    ax.set_xlabel("Ålder")
    ax.set_ylabel("Systoliskt blodtryck")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# Simulering av sjukdom
def simulate_disease(df, n=1000, seed=42):
    p = df["disease"].mean()              
    np.random.seed(seed)
    simulated = np.random.binomial(1, p, n)
    return simulated.mean()

# Konfidensintervall för systoliskt blodtryck
def bp_ci(df):
    bp = df['systolic_bp'].dropna()
    mean_bp = bp.mean()
    se = bp.std(ddof=1) / len(bp)**0.5
    ci_lower = mean_bp - 1.96 * se
    ci_upper = mean_bp + 1.96 * se
    return ci_lower, ci_upper

# Hypotesprövning rökare vs icke-rökare
def ttest_smokers(df):
    df['smoker'] = df['smoker'].str.strip().str.lower()
    df_clean = df[['systolic_bp','smoker']].dropna()
    bp_smokers = df_clean[df_clean['smoker']=='yes']['systolic_bp']
    bp_nonsmokers = df_clean[df_clean['smoker']=='no']['systolic_bp']
    t_stat, p_value = stats.ttest_ind(bp_smokers, bp_nonsmokers, equal_var=False)
    return t_stat, p_value, len(bp_smokers), len(bp_nonsmokers)

# Histogram för systoliskt blodtryck – rökare vs icke-rökare
def plot_bp_hist_smoker_groups(df):
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

# Enkel linjär regression
def simple_regression(df):
    df_clean = df.dropna(subset=['age','weight','systolic_bp'])
    x = df_clean[['age','weight']]
    y = df_clean['systolic_bp']
    model = LinearRegression()
    model.fit(x,y)
    return model

# Regression: scatter + prediktionslinje

def plot_regression(df, model):
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


