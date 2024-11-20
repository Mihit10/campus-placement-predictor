import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def getGraphs():
    dataset = pd.read_csv("Optimized_Placement_Dataset_with_Noise.csv")
    yes_no_columns = [
        'Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript', 
        'Knows HTML', 'Knows CSS', 'Knows Cricket', 'Knows Dance', 
        'Participated in College Fest', 'Was in Coding Club'
    ]
    dataset[yes_no_columns] = dataset[yes_no_columns].replace({'Yes': 1, 'No': 0})
    # Drop unnecessary columns if any
    if 'Name of Student' in dataset.columns and 'Roll No.' in dataset.columns:
        dataset = dataset.drop(columns=['Name of Student', 'Roll No.'])
    # Remove outliers based on Placement Package
    Q1 = dataset['Placement Package'].quantile(0.25)
    Q3 = dataset['Placement Package'].quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset['Placement Package'] < (Q1 - 1.5 * IQR)) | (dataset['Placement Package'] > (Q3 + 1.5 * IQR)))]


    #heatmaps 
    plt.figure(figsize=(10, 6))
    correlation_matrix = dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()

    # Scatterplot for CGPA vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="CGPA", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: CGPA vs Placement Package")
    plt.xlabel("CGPA")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_cgpa_vs_package.png")
    plt.close()


    # Scatterplot for No. of DSA questions vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="No. of DSA questions", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: No. of DSA Questions vs Placement Package")
    plt.xlabel("No. of DSA Questions")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_dsa_vs_package.png")
    plt.close()

    # Scatterplot: No. of Backlogs vs Placement Package
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset, x="No. of backlogs", y="Placement Package", alpha=0.6)
    plt.title("Scatterplot: No. of Backlogs vs Placement Package")
    plt.xlabel("No. of Backlogs")
    plt.ylabel("Placement Package")
    plt.savefig("scatterplot_backlogs_vs_package.png")
    plt.close()


    # Boxplot: Branch of Engineering vs Placement Package
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset, x="Branch of Engineering", y="Placement Package")
    plt.title("Boxplot: Branch of Engineering vs Placement Package")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_branch_vs_package.png")
    plt.close()

    # Countplot: Students in Coding Club by Branch of Engineering
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataset, x="Was in Coding Club", hue="Branch of Engineering")
    plt.title("Countplot: Students in Coding Club by Branch of Engineering")
    plt.xlabel("Was in Coding Club (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.legend(title="Branch", loc="upper right")
    plt.tight_layout()
    plt.savefig("countplot_coding_club_by_branch.png")
    plt.close()

if __name__ == "__main__":
    getGraphs()