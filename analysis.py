"""
Analysis notebook-style script for plotting evaluation results.
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

def load_results(csv_file: str = "evaluation_results.csv") -> pd.DataFrame:
    """Load evaluation results from CSV."""
    return pd.read_csv(csv_file)

def plot_results(df: pd.DataFrame):
    """Create plots comparing A/B test results."""
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean Scores
    scores_by_prompt = df.groupby('prompt_version')['score'].agg(['mean', 'std'])
    
    ax1.bar(scores_by_prompt.index, scores_by_prompt['mean'], 
            yerr=scores_by_prompt['std'], capsize=5, alpha=0.7,
            color=['#2E86AB', '#A23B72'])
    ax1.set_title('Mean Response Quality Score')
    ax1.set_ylabel('Score (1-5)')
    ax1.set_xlabel('System Prompt')
    ax1.set_ylim(0, 5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(scores_by_prompt.iterrows()):
        ax1.text(i, row['mean'] + 0.1, f'{row["mean"]:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Mean Latency
    latency_by_prompt = df.groupby('prompt_version')['latency_ms'].agg(['mean', 'std'])
    
    ax2.bar(latency_by_prompt.index, latency_by_prompt['mean'],
            yerr=latency_by_prompt['std'], capsize=5, alpha=0.7,
            color=['#2E86AB', '#A23B72'])
    ax2.set_title('Mean Response Latency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_xlabel('System Prompt')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(latency_by_prompt.iterrows()):
        ax2.text(i, row['mean'] + row['std'] + 50, f'{row["mean"]:.0f}ms', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'evaluation_results.png'")

def print_detailed_analysis(df: pd.DataFrame):
    """Print detailed statistical analysis."""
    print("=== Detailed Analysis ===\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(df.groupby('prompt_version').agg({
        'score': ['count', 'mean', 'std', 'min', 'max'],
        'latency_ms': ['mean', 'std', 'min', 'max']
    }).round(2))
    
    print("\n" + "="*50)
    
    # Score distribution
    print("\nScore Distribution:")
    score_dist = df.groupby(['prompt_version', 'score']).size().unstack(fill_value=0)
    print(score_dist)
    
    print("\n" + "="*50)
    
    # Best and worst performing queries
    print("\nBest Performing Queries (Prompt A):")
    best_a = df[df['prompt_version'] == 'A'].nlargest(3, 'score')[['query', 'score']]
    for _, row in best_a.iterrows():
        print(f"  Score {row['score']}: {row['query']}")
    
    print("\nWorst Performing Queries (Prompt A):")
    worst_a = df[df['prompt_version'] == 'A'].nsmallest(3, 'score')[['query', 'score']]
    for _, row in worst_a.iterrows():
        print(f"  Score {row['score']}: {row['query']}")

def main():
    """Main analysis function."""
    try:
        df = load_results()
        print(f"Loaded {len(df)} results")
        
        plot_results(df)
        print_detailed_analysis(df)
        
    except FileNotFoundError:
        print("evaluation_results.csv not found. Run main.py first!")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
