import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from scipy.stats import shapiro, kurtosis, chi2_contingency, ttest_ind, mannwhitneyu, kendalltau, spearmanr, pearsonr
from scipy.stats.contingency import association
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm


def load_data_from_file(data_path: str) -> pd.DataFrame:
    '''
    Load data from a CSV file at the specified path.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame or None if an error occurs.
    '''
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            print(f"Data loaded successfully from: {data_path}")
            return data
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            return None
    else:
        print(f"Path does not exist: {data_path}")
        return None


def save_csv_data_to_file(data: pd.DataFrame, data_path: str, overwrite: bool = False) -> bool:
    '''
    Save a DataFrame to a CSV file at the specified path.

    Args:
        data (pd.DataFrame): DataFrame to save.
        data_path (str): Destination path for the CSV file.
        overwrite (bool, optional): Whether to overwrite if file exists. Defaults to False.

    Returns:
        bool: True if file was saved successfully, False otherwise.
    '''
    if not isinstance(data, pd.DataFrame) or data.empty:
        print("Provided data is empty or not a valid DataFrame. Nothing to save.")
        return False

    if os.path.exists(data_path) and not overwrite:
        print(f"File already exists at: {data_path}. Set overwrite=True to replace it.")
        return False

    try:
        data.to_csv(data_path, index=False)
        print(f"Data saved successfully at: {data_path}")
        return True
    except Exception as e:
        print(f"Error saving data to {data_path}: {e}")
        return False


def check_missing_values(data:pd.DataFrame) -> pd.Series:
    '''
    Check for missing values in a DataFrame and print a summary.

    Args:
        data (pd.DataFrame): DataFrame to check for missing values.

    Returns:
        pd.Series: Series with counts of missing values per column.
    '''
    if data is None:
        print("No data provided to check for missing values.")
        return None

    print("Checking for missing values in the data...")

    try:
        missing_values = data.isnull().sum()

        if missing_values.sum() == 0:
            print("No missing values found in the data.")
        else:
            print("Missing values found in the data:")
            print(missing_values[missing_values > 0])

        return missing_values  # Return the full series for further use if needed

    except Exception as e:
        print(f"Error while checking for missing values: {e}")
        return None


def check_duplicates(data: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    '''
    Check for duplicate rows in a DataFrame and print a summary.

    Args:
        data (pd.DataFrame): DataFrame to check for duplicates.

    Returns:
        pd.DataFrame: DataFrame containing duplicate rows, if any; empty DataFrame if none.
    '''
    if data is None:
        print("No data provided to check for duplicates.")
        return None

    print("Checking for duplicate rows in the data...")

    try:
        # Identify duplicate rows
        duplicates = data[data.duplicated(keep=False)]  # keep=False marks all duplicates

        if duplicates.empty:
            print("No duplicate rows found in the data.")
        else:
            print(f"Found {len(duplicates)} duplicate rows in the data.")
            print("Here are the duplicate rows:")
            display(duplicates)  # display() is handy in Jupyter Notebooks, use print(duplicates) if running as script

        if drop_duplicates:
            data.drop_duplicates(inplace=True)
            print("Dropped duplicate rows from the data.")

        return duplicates  # Return duplicates DataFrame for further inspection if needed

    except Exception as e:
        print(f"Error while checking for duplicates: {e}")
        return None



def plot_numerical_data_with_iqr(data: pd.DataFrame, features: list, scale_percent_cols: list = None):
    '''
    Plot boxplots with stripplots for numerical data, including IQR outlier bounds.

    Args:
        data (pd.DataFrame): The dataset containing numerical columns.
        features (list): List of feature names to include in the plot.
        scale_percent_cols (list, optional): Columns to scale from 0-1 to 0-100 (e.g., percent values).

    Returns:
        None (displays the plot)
    '''

    # --- Prepare and scale data ---
    numerical_data = data[features].copy()

    # Scale percent columns if provided
    if scale_percent_cols:
        for col in scale_percent_cols:
            if col in numerical_data.columns:
                numerical_data[col] = numerical_data[col] * 100

    # Melt data for stripplot and coloring
    numerical_melted = numerical_data.melt(var_name='Feature', value_name='Value')

    # --- Set up the plot ---
    plt.figure(figsize=(15, 10))

    # Transparent boxplots with gray outlines
    ax = sns.boxplot(
        data=numerical_data,
        color='gray',
        showcaps=True,
        boxprops={'facecolor': 'none', 'edgecolor': 'gray', 'linewidth': 2},
        whiskerprops={'color': 'gray', 'linewidth': 2},
        capprops={'color': 'gray', 'linewidth': 2},
        medianprops={'color': 'black', 'linewidth': 2},
        showfliers=False  # Outliers handled via stripplot
    )

    # Stripplot colored by value
    strip = sns.stripplot(
        data=numerical_melted,
        x='Feature', y='Value',
        hue='Value',
        palette='coolwarm',
        size=8,
        jitter=True,
        alpha=0.8,
        ax=ax
    )

    # --- IQR calculation ---
    iqr_data = {}
    for col in numerical_data.columns:
        Q1 = numerical_data[col].quantile(0.25)
        Q3 = numerical_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_data[col] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'lower': lower_bound, 'upper': upper_bound}

    # --- Plot IQR bounds as dashed red lines ---
    for i, feature in enumerate(numerical_data.columns):
        bounds = iqr_data[feature]
        # Lower bound
        ax.hlines(bounds['lower'], i - 0.4, i + 0.4, colors='red', linestyles='dashed', linewidth=2, label='IQR Bound' if i == 0 else "")
        # Upper bound
        ax.hlines(bounds['upper'], i - 0.4, i + 0.4, colors='red', linestyles='dashed', linewidth=2)

    # --- Colorbar for stripplot values ---
    strip.legend_.remove()  # Remove default stripplot legend
    norm = plt.Normalize(numerical_melted['Value'].min(), numerical_melted['Value'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Value (Scaled % & Scores)', fontsize=12)

    # --- Plot styling ---
    plt.title("Boxplot of Numerical Candy Data with Value-based Stripplot and IQR Outlier Thresholds", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.ylabel("Value (Scaled Percentages & Scores)", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.tight_layout()
    
    plt.show()


def check_normal_distribution(data, columns):
    for column in columns:
        stat, p_value = shapiro(data[column])
        print(f'{column}: W={stat:.3f}, p-value={p_value:.3f}')
        if p_value > 0.05:
            print(f"→ {column} looks normally distributed (fail to reject H0)")
        else:
            print(f"→ {column} is NOT normally distributed (reject H0)")




def check_skewness_and_kurtosis(data, columns):
    """
    Calculates skewness and kurtosis for specified columns, focusing on key interpretations.
    """

    for column in columns:
        skew = data[column].skew()
        kurt = kurtosis(data[column], fisher=True)

        # Skewness Interpretation
        if skew > 0:
            skew_desc = "Right (positively) skewed: More data points on the left, tail extending to the right."
        elif skew < 0:
            skew_desc = "Left (negatively) skewed: More data points on the right, tail extending to the left."
        else:
            skew_desc = "Approximately symmetric."

        # Kurtosis Interpretation
        if kurt > 1: # use just 1, to simplify the result.
            kurt_desc = "Leptokurtic (heavy tails): More outliers, sharper peak."
        elif kurt < -1: # use just -1, to simplify the result.
            kurt_desc = "Platykurtic (light tails): Fewer outliers, flatter peak."
        else:
            kurt_desc = "Mesokurtic (normal tails): Outliers and peak similar to a normal distribution."

        # Print Simplified Explanation
        print(f"{column}:")
        print(f"Skewness ({skew}): {skew_desc}")
        print(f"Kurtosis ({kurt}): {kurt_desc}")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(data[column], kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {column}")
        sns.boxplot(data[column], ax=axes[1])
        axes[1].set_title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.show()


def plot_distribution(data, columns):
    for column in columns:
        sns.displot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.axvline(data[column].mean(), color='r', linestyle='--', label='mean')
        plt.axvline(data[column].median(), color='g', linestyle='-', label='median')
        plt.show()

def plot_frequency(data, ingredients, name="Ingredients"):
    frequency = data[ingredients].sum()
    frequency = frequency.sort_values(ascending=False)
    frequency = frequency / frequency.sum()

    ax = frequency.plot(kind='bar')

    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.title(f"Frequency of {name}")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels
    plt.show()



def plot_frequency_combinations(data, top_n=10, column='ingredient_combinations', name='Candy Ingrediengts'):  # Added top_n parameter
    frequency = data[column].value_counts()
    frequency = frequency / frequency.sum()

    # Filter top N combinations
    if top_n:
        frequency = frequency.head(top_n)

    ax = frequency.plot(kind='barh', figsize=(10, 6))  # Horizontal bar plot

    # Format y-axis as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

    if top_n:
        plt.title(f"Top {top_n} Ingredient Combinations")
    else:
        plt.title(f"{name} Combinations")

    plt.xlabel("Percentage")
    plt.ylabel(f"{name} Combinations")
    plt.tight_layout() # prevents labels from being cut off.
    plt.show()


def plot_form_and_ingredients_heatmap_with_totals(prepared_data):
    """
    Plots a heatmap of form and ingredient combinations with total counts on axes.
    """

    # Create cross-tabulation
    cross_tab = pd.crosstab(
        prepared_data['form_combinations'],
        prepared_data['ingredient_combinations']
    )

    # Calculate row and column totals
    row_totals = cross_tab.sum(axis=1)
    col_totals = cross_tab.sum(axis=0)

    # Modify axis labels with totals
    row_labels = [f"{label} ({total})" for label, total in row_totals.items()]
    col_labels = [f"{label} ({total})" for label, total in col_totals.items()]

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(cross_tab, annot=True, cmap='viridis', fmt='d',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title('Candy Form and Ingredient Combinations with Totals')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_composition_correlation(df, composition_columns, winpercent_column='winpercent', figsize=(8, 6), method='spearman', significance_level=0.05):
    """
    Plots a heatmap of the correlation matrix between composition columns and winpercent.
    """
    # Compute correlation matrix
    corr_matrix = df[composition_columns + [winpercent_column]].corr(method=method)

    # Mask for upper triangle to avoid duplicate information
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Calculate p-values for all correlations
    p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if method == 'spearman':
                _, p_value = spearmanr(df[col1], df[col2], nan_policy='omit')
            elif method == 'pearson':
                _, p_value = pearsonr(df[col1], df[col2])
            elif method == 'kendall':
                _, p_value = kendalltau(df[col1], df[col2], nan_policy='omit')
            else:
                raise ValueError("Method must be 'spearman', 'pearson', or 'kendall'")
            p_matrix.loc[col1, col2] = p_value

    # Prepare annotations with significance markers
    annotations = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            corr_value = corr_matrix.iloc[i, j]
            p_value = p_matrix.iloc[i, j]
            annotation = f"{corr_value:.2f}" # format corr_value before adding the star
            if p_value < significance_level:
                annotation += "*"  # Add asterisk for significance
            annotations[i, j] = annotation

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annotations,
        fmt='',  # Empty to use custom annotations
        cmap='coolwarm',
        center=0,
        mask=mask,
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Correlation Between Candy Composition and Winpercent', fontsize=14, weight='bold', pad=15)
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.show()

    print(corr_matrix)


def cramers_v_and_p(df, col1, col2, correction=False):
    """
    Computes Cramér's V and p-value between two categorical columns.
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, _, _ = chi2_contingency(contingency)
    cramers_v = association(contingency, method='cramer', correction=correction)
    return cramers_v, p

def cramers_v_matrix_with_p(df, rows, cols, correction=True):
    """
    Builds matrices of Cramér's V and p-values.
    """
    cramer_matrix = pd.DataFrame(index=rows, columns=cols)
    p_matrix = pd.DataFrame(index=rows, columns=cols)
    for row in rows:
        for col in cols:
            cramers_v, p = cramers_v_and_p(df, row, col, correction=correction)
            cramer_matrix.loc[row, col] = cramers_v
            p_matrix.loc[row, col] = p
    return cramer_matrix.astype(float), p_matrix.astype(float)

def categorical_corr_with_significance(raw_data, X_COLUMNS, Y_COLUMNS, name="Ingredients", significance_level=0.05, mask = True):
    """
    Calculates and visualizes Cramér's V with significance markers.
    """
    cramer_matrix, p_matrix = cramers_v_matrix_with_p(raw_data, X_COLUMNS, Y_COLUMNS, correction=True)

    # Prepare annotations with significance markers
    annotations = np.empty_like(cramer_matrix, dtype=object)
    for i in range(cramer_matrix.shape[0]):
        for j in range(cramer_matrix.shape[1]):
            cramer_value = cramer_matrix.iloc[i, j]
            p_value = p_matrix.iloc[i, j]
            annotation = f"{cramer_value:.2f}"
            if p_value < significance_level:
                annotation += "*"  # Add asterisk for significance
            annotations[i, j] = annotation

    # Plot Cramer's V heatmap with annotations
    plt.figure(figsize=(12, 8))
    if mask:
        mask = np.triu(np.ones_like(cramer_matrix, dtype=bool))
        sns.heatmap(cramer_matrix, annot=annotations, fmt='', cmap='coolwarm', mask=mask, square=True)  # fmt='' to use custom annotations
    else:
        sns.heatmap(cramer_matrix, annot=annotations, fmt='', cmap='coolwarm', square=True)
    plt.title(f"Cramér's V for {name} (Significance at p < {significance_level})", fontsize=16)
    plt.xlabel(f"{name}", fontsize=12)
    plt.ylabel(f"{name}", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_frequency_vs_winpercent_box(df, ingredient_columns, winpercent_column='winpercent', figsize=(14, 8)):
    """
    Plots a side-by-side visualization:
    - Left: Barplot of how often each ingredient appears.
    - Right: Violin plot showing the distribution of winpercent for each ingredient.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Your data containing ingredient columns and winpercent.
    ingredient_columns : list of str
        List of ingredient column names (binary 0/1 columns).
    winpercent_column : str
        Column name containing winpercent values.
    figsize : tuple
        Size of the overall figure.
    """
    # Prepare data for frequency
    freq_data = df[ingredient_columns].sum().reset_index()
    freq_data.columns = ['ingredient', 'count']
    
    # Prepare data for winpercent (melted to long form)
    melted_data = pd.melt(
        df, 
        id_vars=[winpercent_column], 
        value_vars=ingredient_columns,
        var_name='ingredient', 
        value_name='has_ingredient'
    )
    
    # Filter only rows where the ingredient is present
    melted_data = melted_data[melted_data['has_ingredient'] == 1]

    # Calculate mean winpercent to sort ingredients for visual alignment
    mean_winpercent = melted_data.groupby('ingredient')[winpercent_column].mean().reset_index()
    sorted_ingredients = mean_winpercent.sort_values(winpercent_column, ascending=False)['ingredient'].tolist()
    
    # --- Set up plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Left: Frequency barplot
    sns.barplot(
        y='ingredient', 
        x='count', 
        data=freq_data, 
        order=sorted_ingredients, 
        ax=ax1, 
        color='lightgray', 
        edgecolor='black'
    )
    ax1.set_title('Ingredient Frequency', fontsize=14, weight='bold')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Ingredient')

    # Right: Violin plot for winpercent distribution
    sns.violinplot(
        y='ingredient',
        x=winpercent_column,
        data=melted_data,
        order=sorted_ingredients,
        ax=ax2,
        palette='coolwarm',
        linewidth=2.5
    )
    ax2.set_title('Winpercent Distribution per Ingredient', fontsize=14, weight='bold')
    ax2.set_xlabel('Winpercent')

    # Capitalize ingredient names for aesthetics
    formatted_labels = [ingredient.capitalize() for ingredient in sorted_ingredients]
    ax1.set_yticklabels(formatted_labels, fontsize=12)
    ax2.set_yticklabels(formatted_labels, fontsize=12)  # Fix: make them visible on right plot too

    plt.tight_layout()
    plt.show()


def calculate_significance_with_tests(df, ingredient_columns, winpercent_column='winpercent', alpha=0.05):
    """
    Calculates significance of winpercent differences between candies with and without each ingredient,
    automatically selecting appropriate statistical test based on normality.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Your dataset.
    ingredient_columns : list of str
        List of binary ingredient columns.
    winpercent_column : str
        Column containing winpercent.
    alpha : float
        Significance level for normality and hypothesis testing.
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with p-values, significance, method used, effect size, sample size.
    """
    
    results = []

    for ingredient in ingredient_columns:
        has_ingr = df[df[ingredient] == 1][winpercent_column]
        no_ingr = df[df[ingredient] == 0][winpercent_column]

        # Sample sizes
        n_with = len(has_ingr)
        n_without = len(no_ingr)

        # Normality test
        p_norm_with = shapiro(has_ingr)[1] if n_with >= 3 else 1  # Avoid Shapiro on too-small samples
        p_norm_without = shapiro(no_ingr)[1] if n_without >= 3 else 1

        normal = (p_norm_with > alpha) and (p_norm_without > alpha)

        # Choose appropriate test
        if normal:
            stat, p_value = ttest_ind(has_ingr, no_ingr, equal_var=False)
            method = 't-test'
            # Calculate Cohen's d for effect size
            mean_diff = has_ingr.mean() - no_ingr.mean()
            pooled_std = np.sqrt((has_ingr.std()**2 + no_ingr.std()**2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
           
            stat, p_value, method, cohen_d = np.nan , np.nan, np.nan, np.nan

        results.append({
            'ingredient': ingredient,
            'significant': p_value < alpha,
            'effect size': round(cohen_d, 3),

            'sample_size_with': n_with,
            'sample_size_without': n_without,
            'p_value': p_value,
            'method': method,
            'normal_dist (both)': normal
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Correct p-values for multiple testing (Holm-Bonferroni correction)
    corrected_p = multipletests(results_df['p_value'], method='holm')[1]
    results_df['p_value_corrected'] = corrected_p
    results_df['significant_corrected'] = corrected_p < alpha

    return results_df





def plot_combination_frequency_vs_winpercent_box(df, ingredient_columns, winpercent_column='winpercent', figsize=(16, 10), top_n=None):
    """
    Beautiful dual-plot visualization for ingredient combinations:
    - Left: Barplot of combination frequency (colored by mean winpercent for emphasis).
    - Right: Boxplot of winpercent distribution (with mean markers).
    """
    
    # Step 1: Create combination column
    def combo(row):
        return '+'.join([feat for feat in ingredient_columns if row[feat] == 1]) or 'None'

    df['combination'] = df.apply(combo, axis=1)

    # Step 2: Frequency and mean winpercent
    combo_counts = df['combination'].value_counts().reset_index()
    combo_counts.columns = ['combination', 'count']
    combo_winpercent = df.groupby('combination')[winpercent_column].mean().reset_index()

    # Step 3: Merge and filter top N
    combo_summary = combo_counts.merge(combo_winpercent, on='combination')
    if top_n:
        combo_summary = combo_summary.sort_values('count', ascending=False).head(top_n)

    # Step 4: Prepare filtered data and sorting
    filtered_df = df[df['combination'].isin(combo_summary['combination'])]
    sorted_combinations = combo_summary.sort_values(winpercent_column, ascending=False)['combination'].tolist()
    

    # Normalize winpercent for coloring
    norm = plt.Normalize(combo_summary['winpercent'].min(), combo_summary['winpercent'].max())
    cmap = plt.cm.cividis
    bar_colors = cmap(norm(combo_summary.set_index('combination').loc[sorted_combinations]['winpercent']))

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    #  Barplot (Frequency with color mapped to winpercent)
    bars = ax1.barh(
        y=sorted_combinations,
        width=combo_summary.set_index('combination').loc[sorted_combinations]['count'],
        color=bar_colors,
        edgecolor='black',
        height=0.6
    )
    ax1.set_title('Ingredient Combination Frequency\n(Color = Mean Winpercent)', fontsize=15, weight='bold', pad=15)
    ax1.set_xlabel('Count', fontsize=13)
    ax1.set_ylabel('Ingredient Combination', fontsize=13)

    # Annotate counts next to bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center', fontsize=11, color='black')

    # Boxplot (Winpercent distribution with mean markers)
    sns.boxplot(
        y='combination',
        x=winpercent_column,
        data=filtered_df,
        order=sorted_combinations,
        ax=ax2,
        width=0.6,
        color='white',
        linewidth=1.5,
        fliersize=4
    )
    
    # Add mean markers to boxplot
    means = filtered_df.groupby('combination')[winpercent_column].mean().reindex(sorted_combinations)
    ax2.scatter(
        means.values,
        range(len(sorted_combinations)),
        color='darkred',
        s=80,
        zorder=3,
        label='Mean Winpercent'
    )
    
    ax2.set_title('Winpercent Distribution per Combination', fontsize=15, weight='bold', pad=15)
    ax2.set_xlabel('Winpercent', fontsize=13)
    ax2.set_ylabel('')

    #  Y-axis labels on both sides for readability
    ax1.set_yticklabels([combo.capitalize() for combo in sorted_combinations], fontsize=12)
    ax2.set_yticklabels([combo.capitalize() for combo in sorted_combinations], fontsize=12)


    #  Style & Layout
    ax1.invert_yaxis()  # Highest performers on top
    plt.tight_layout()
    plt.show()


def calculate_combination_significance(df, ingredient_columns, winpercent_column='winpercent', alpha=0.05, drop_na_rows=True):
    """
    Calculates significance of winpercent differences between candies with and without each ingredient combination,
    selecting appropriate statistical test based on normality.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Dataset.
    ingredient_columns : list of str
        List of binary ingredient columns.
    winpercent_column : str
        Target variable.
    alpha : float
        Significance level.
        
    Returns:
    --------
    pd.DataFrame
        Summary table with significance, effect sizes, and tests used.
    """

    results = []

    # Create combination column
    def combo(row):
        return '+'.join([feat for feat in ingredient_columns if row[feat] == 1]) or 'None'

    df['combination'] = df.apply(combo, axis=1)

    # Get unique combinations
    unique_combinations = df['combination'].unique()

    for combination in unique_combinations:
        has_combo = df[df['combination'] == combination][winpercent_column]
        no_combo = df[df['combination'] != combination][winpercent_column]

        n_with = len(has_combo)
        n_without = len(no_combo)

        # Skip if sample too small
        if n_with < 3 or n_without < 3:
            p_value = np.nan
            method = 'Too small sample'
            cohen_d = np.nan
            normal = False
        else:
            # Normality test
            p_norm_with = shapiro(has_combo)[1]
            p_norm_without = shapiro(no_combo)[1]
            normal = (p_norm_with > alpha) and (p_norm_without > alpha)

            # Choose test
            if normal:
                stat, p_value = ttest_ind(has_combo, no_combo, equal_var=False)
                method = 't-test'
                # Effect size
                mean_diff = has_combo.mean() - no_combo.mean()
                pooled_std = np.sqrt((has_combo.std() ** 2 + no_combo.std() ** 2) / 2)
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            else:
                stat, p_value = mannwhitneyu(has_combo, no_combo, alternative='two-sided')
                method = 'Mann-Whitney U'
                # here the rank-biserial correlation coefficient (r) should be used as effect size
                rb_corr = 1 - (2 * stat) / (n_with * n_without)
                cohen_d = rb_corr

        # Append results
        results.append({
            'combination': combination,
            'significant': p_value < alpha if not np.isnan(p_value) else False,
            'effect size': round(cohen_d, 3) if not np.isnan(cohen_d) else np.nan,
            'sample_size_with': n_with,
            'sample_size_without': n_without,
            'p_value': p_value,
            'method': method,
            'normal_dist (both)': normal,
            'effect_direction': 'With > Without' if cohen_d > 0 else ('Without > With' if cohen_d < 0 else 'Neutral')
        })

    # Final results DataFrame
    results_df = pd.DataFrame(results)

    # Multiple testing correction
    valid_p_values = results_df['p_value'].dropna()
    corrected_p_values = multipletests(valid_p_values, method='holm')[1]
    
    # Merge corrected p-values back into DataFrame
    results_df.loc[results_df['p_value'].notna(), 'p_value_corrected'] = corrected_p_values
    results_df['significant_corrected'] = results_df['p_value_corrected'] < alpha

    results_df.sort_values(by="p_value_corrected", ascending=True, inplace=True)

    # drop rows with NaN 
    if drop_na_rows:
        results_df.dropna(inplace=True)
    

    return results_df


def calculate_vif(data, features):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in a dataset.
    
    Parameters:
    ----------
    data : pd.DataFrame
        The dataset containing the features.
    features : list of str
        List of feature column names.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names as index and VIF as values.
    """
    # Create a DataFrame to store the results
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(data[features].values, i) for i in range(len(features))]

    # mark features with high VIF
    vif_data["High VIF"] = vif_data["VIF"] > 5
    

    return vif_data

def build_ols_model_formula(data, formula):
    """
    Builds an OLS regression model using a formula.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        formula (str): The formula specifying the model.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted model.
    """
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    return model

def plot_cumulative_coefficients(model, significance_level=0.05, figsize=(10, 6)):
    """
    Plots stacked (cumulative) coefficients of significant features from a fitted OLS model.
    
    Parameters:
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model.
    significance_level : float, optional (default=0.05)
        Significance level to filter coefficients.
    figsize : tuple, optional (default=(10, 6))
        Size of the plot.
    """
    # Extract coefficients, standard errors, and p-values (drop intercept)
    coefs = model.params.drop('Intercept')
    errors = model.bse.drop('Intercept')
    p_values = model.pvalues.drop('Intercept')
    
    # Filter for significant coefficients
    significant = p_values[p_values < significance_level].index
    coefs = coefs.loc[significant]
    errors = errors.loc[significant]

    # Sort by absolute value for better visualization
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=True).index)
    errors = errors.reindex(coefs.index)

    # Prepare DataFrame
    coef_df = pd.DataFrame({'Coefficient': coefs})
    coef_df['Cumulative'] = coef_df['Coefficient'].cumsum()
    coef_df['Error'] = errors

    coef_df = coef_df.reset_index().rename(columns={'index': 'Feature'})

    # Color coding: green for positive, red for negative
    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]

    # Plot cumulative effect
    plt.figure(figsize=figsize)
    plt.step(coef_df['Feature'], coef_df['Cumulative'], where='mid', color='blue', linewidth=3, marker='o')

    # Add data labels
    for i, row in coef_df.iterrows():
        plt.text(row['Feature'], row['Cumulative'] + 0.01, f"{row['Cumulative']:.2f}", 
                 ha='center', va='bottom', fontsize=10)

    # Bar plot as background to show coefficient impact size
    plt.bar(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.3, label='Effect size')

    # Optional: Add error bars (confidence intervals)
    plt.errorbar(coef_df['Feature'], coef_df['Coefficient'], yerr=1.96*coef_df['Error'], 
                 fmt='o', color='black', capsize=5, label='95% CI')

    # Plot settings
    plt.title('Cumulative Impact of Significant Features on Winpercent')
    plt.ylabel('Cumulative Effect on Winpercent')
    plt.xlabel('Feature')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # zero line
    plt.legend()
    plt.show()

# diagnose the model

def qqplot(model):
    """
    Creates a QQ-plot and residuals plot for a given OLS model.
    
    Parameters:
    ----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS model.
    """
    # QQ-plot
    sm.qqplot(model.resid, line='s')
    plt.title("QQ-Plot of Residuals")
    plt.show()


def plot_ols_coefficients(params, p_values, significance_level=0.05, marginal_significance_level=0.1, figsize=(10, 6)):
    """
    Plots the coefficients of an OLS regression model as a bar plot,
    indicating significant coefficients with a star, and intuitive colors.
    """

    coefficients = params.drop('Intercept')
    p_values = p_values.drop('Intercept')

    coef_df = pd.DataFrame({
        'Coefficient': coefficients,
        'P-value': p_values
    })
    coef_df['Highly_Significant'] = coef_df['P-value'] < 0.01
    coef_df['Significant'] = (coef_df['P-value'] < significance_level) & (~coef_df['Highly_Significant'])
    coef_df['Marginal_Significant'] = (coef_df['P-value'] >= significance_level) & (coef_df['P-value'] < marginal_significance_level)
    coef_df['Non_Significant'] = (coef_df['P-value'] >= marginal_significance_level)

    coef_df['Annotation'] = coef_df['Coefficient'].round(3).astype(str)
    coef_df.loc[coef_df['Significant'], 'Annotation'] += '**'
    coef_df.loc[coef_df['Marginal_Significant'], 'Annotation'] += '*'
    coef_df.loc[coef_df['Highly_Significant'], 'Annotation'] += '***'

    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    # Create a color palette based on coefficient sign and significance
    colors = []
    for coef, sig, marg_sig, high_sig in zip(coef_df['Coefficient'], coef_df['Significant'], coef_df['Marginal_Significant'], coef_df['Highly_Significant']):
        if high_sig:
            if coef > 0:
                colors.append('limegreen')  # Green for significant positive
            else:
                colors.append('salmon')  # Red for significant negative
        elif sig:
            if coef > 0:
                colors.append('lightgreen')  # Light green for marginal significant positive
            else:
                colors.append('lightsalmon')  # Light red for marginal significant negative
        elif marg_sig:
            if coef > 0:
                colors.append('palegreen')
            else:
                colors.append('lightcoral')
        else:
            colors.append('lightgray')  # Gray for non-significant

    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Coefficient', y=coef_df.index, data=coef_df, hue =coef_df.index,palette=colors, edgecolor='none', legend=False)

    for index, value in enumerate(coef_df['Coefficient']):
        plt.text(value + (0.01 if value >= 0 else -0.05), index, coef_df['Annotation'].iloc[index], va='center')

    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('OLS Regression Coefficients with Significance', fontsize=16)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Predictor Variables', fontsize=12)

    max_coef = coef_df['Coefficient'].abs().max()
    ax.set_xlim(-max_coef * 1.5, max_coef * 1.5)

    # Add dashed outlines for significant bars
    for idx, (index, sig, marg_sig) in enumerate(zip(coef_df.index, coef_df['Significant'], coef_df['Marginal_Significant'])):
        if sig:
            bar = ax.patches[idx]
            bar.set_edgecolor('black')
            bar.set_linewidth(1)
            bar.set_linestyle('--')
        elif marg_sig:
            bar = ax.patches[idx]
            bar.set_edgecolor('black')
            bar.set_linewidth(1)

    # Add legend
    plt.legend(['*** p < 0.01', '** p < 0.05', '* p < 0.1'])

    plt.tight_layout()
    plt.show()



def visualize_sugar_effect(df, model_results, sugar_col='sugarpercent', win_col='winpercent', non_linear_col='sugarpercent_squared', non_liner_factor=2):
    """
    Visualizes the non-linear effect of sugarpercent on winpercent.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=sugar_col, y=win_col, data=df)

    sugar_range = np.linspace(df[sugar_col].min(), df[sugar_col].max(), 100)
    X_pred = pd.DataFrame({non_linear_col: sugar_range**non_liner_factor})

    # Add other columns with their mean values from the original DataFrame
    for col in df.columns:
        if col not in [sugar_col, win_col, non_linear_col]:
            X_pred[col] = df[col].mean()

    X_pred = sm.add_constant(X_pred)
    y_pred = model_results.predict(X_pred)

    plt.plot(sugar_range, y_pred, color='red', linewidth=2, label='Fitted Curve')

    plt.title('Sugar Percent vs. Win Percent with Fitted Curve', fontsize=16)
    plt.xlabel('Sugar Percent', fontsize=12)
    plt.ylabel('Win Percent', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()