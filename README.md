# Clustering-based CoT Example Selection Experiment

## Project Overview

This project contains code for comparative experiments on clustering-based Chain-of-Thought (CoT) example selection methods to analyze differences in reasoning accuracy, token efficiency, and diversity compared to MMR-based and random selection methods.

## Experiment Setup

1. **Datasets**: GSM8K, CommonsenseQA, ARC (Combined dataset with 1500 examples total)
2. **Embedding Model**: SentenceTransformer('all-mpnet-base-v2')
3. **Example Selection Methods**:
   - Zero-shot (no examples)
   - Random selection
   - Similarity-based selection
   - MMR-based (λ=0.7)
   - Clustering-based (KMeans, k=5 / HDBSCAN)
   - CM-MMR (Clustering + MMR + Dynamic Mapping, k=5, λ=0.7)
   - CM-HDBSCAN-MMR (HDBSCAN + MMR + Dynamic Mapping)
4. **LLM**: OpenAI GPT-4o-mini
5. **Evaluation Metrics**: Accuracy, Selection Latency, Diversity Score

## Installation

```bash
# Clone repository
git clone <repository-url>
cd data-mining-report

# Install required packages
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Set your OpenAI API key in the .env file
```

## Usage

### Basic Experiments

```bash
# Single experiment
python src/main.py --dataset combined --method kmeans --num_examples 5

# Repeated experiments (for statistical analysis)
python src/main.py --dataset combined --method random --num_examples 5 --num_repeats 10

# Zero-shot experiment
python src/main.py --dataset combined --method random --num_examples 0 --experiment_name zero
```

### Analysis and Visualization

```bash
# Clustering evaluation with metrics
python scripts/clustering_evaluation.py --dataset combined

# Generate accuracy and latency comparison plots
python scripts/combined_plot.py

# Calculate diversity scores
python scripts/calculate_diversity.py

# Domain-specific accuracy analysis
python scripts/domain_accuracy_comparison.py
```

### Clustering Visualization

```bash
# K-means clustering visualization
python scripts/clustering_visualization.py

# HDBSCAN clustering visualization  
python scripts/hdbscan_visualization.py

# Elbow method for optimal k selection
python scripts/elbow_method.py
```

## Key Features

### 1. Multiple Example Selection Methods
- **Zero-shot**: No examples provided
- **Random**: Random selection from example pool
- **Similarity**: Most similar examples to query
- **MMR**: Maximal Marginal Relevance for diversity
- **K-means**: Cluster-based selection with K-means
- **HDBSCAN**: Density-based clustering selection
- **CM-MMR**: Combined clustering and MMR approach

### 2. Comprehensive Evaluation
- **Accuracy**: Correctness of LLM responses
- **Selection Latency**: Time taken for example selection
- **Diversity Score**: Semantic diversity of selected examples
- **Statistical Analysis**: Multiple runs with min/max/avg statistics

### 3. Advanced Clustering Analysis
- **Silhouette Score**: Cluster quality measurement
- **Davies-Bouldin Index**: Cluster separation evaluation
- **Calinski-Harabasz Score**: Cluster validity assessment
- **Visualization**: 2D/3D UMAP and t-SNE projections

### 4. Multi-Dataset Support
- **GSM8K**: Grade school math problems
- **CommonsenseQA**: Common sense reasoning
- **ARC**: AI2 Reasoning Challenge
- **Combined**: Unified dataset for cross-domain evaluation

## Project Structure

```
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── examples/              # Example pools
├── results/                   # Experiment results
├── analysis/                  # Analysis results (CSV files)
│   ├── clustering_metrics.csv
│   └── diversity_scores.csv
├── plots/                     # Generated visualizations
│   ├── combined_accuracy_latency.png
│   ├── clustering_visualization.png
│   ├── diversity_score_by_method.png
│   └── domain_accuracy_comparison.png
├── scripts/                   # Analysis and utility scripts
│   ├── clustering_evaluation.py
│   ├── combined_plot.py
│   ├── calculate_diversity.py
│   └── domain_accuracy_comparison.py
├── src/
│   ├── data/                  # Data processing code
│   ├── example_selection/     # Example selection implementations
│   ├── llm/                   # GPT API integration
│   ├── analysis/              # Result analysis code
│   ├── utils/                 # Utility functions
│   ├── main.py                # Main execution script
│   └── run_all_experiments.py # Batch experiment runner
├── notebooks/                 # Jupyter notebooks for exploration
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Experimental Results

### Performance Comparison
- **Best Accuracy**: CM-MMR(HDBSCAN) - 86.33%
- **Fastest Selection**: Zero-shot - 0.017ms
- **Most Diverse**: MMR-based methods
- **Most Stable**: K-means clustering

### Key Findings
1. **Clustering methods** show competitive accuracy with traditional approaches
2. **CM-MMR combinations** achieve the highest accuracy scores
3. **Selection latency** varies significantly between methods (0.017ms - 7.5ms)
4. **Zero-shot** performance is surprisingly competitive at 84.17%

## Dependencies

- Python 3.8+
- sentence-transformers
- scikit-learn
- hdbscan
- matplotlib
- numpy
- pandas
- openai
- python-dotenv
