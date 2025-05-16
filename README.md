# Clustering-based CoT Example Selection Experiment

## Project Overview

This project contains code for comparative experiments on clustering-based Chain-of-Thought (CoT) example selection methods to analyze differences in reasoning accuracy, token efficiency, and diversity compared to MMR-based and random selection methods.

## Experiment Setup

1. **Datasets**: GSM8K (100 questions), CommonsenseQA (100 questions)
2. **Embedding Model**: SentenceTransformer('all-mpnet-base-v2')
3. **Example Selection Methods**:
   - Clustering-based (KMeans, k=5 / HDBSCAN)
   - MMR-based (λ=0.7)
   - Random (5 random selections)
   - CM-MMR (Clustering + MMR + Dynamic Mapping, k=5, λ=0.7)
4. **LLM**: OpenAI GPT-4.1 mini

## Installation

```bash
# Clone repository
git clone <repository-url>
cd cot-example-selection

# Install required packages
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Set your OpenAI API key in the .env file
```

## Usage

```bash
# Prepare data
python src/data/prepare_datasets.py

# Run experiments
python src/main.py --dataset gsm8k --method kmeans --num_examples 5
python src/main.py --dataset commonsenseqa --method mmr --num_examples 5
python src/main.py --dataset gsm8k --method random --num_examples 5
python src/main.py --dataset commonsenseqa --method cm-mmr --num_examples 5

# Or run all experiments at once
python src/run_all_experiments.py

# Analyze results
python src/analysis/analyze_results.py
```

## Project Structure

```
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed datasets
│   └── examples/              # Example pools
├── results/                   # Experiment results
├── src/
│   ├── data/                  # Data processing code
│   ├── example_selection/     # Example selection method implementations
│   ├── llm/                   # GPT API call related code
│   ├── analysis/              # Result analysis code
│   ├── utils/                 # Utility functions
│   ├── main.py                # Main execution script
│   └── run_all_experiments.py # Script to run all experiments
├── .env.example               # Environment variable example
├── requirements.txt           # Required package list
└── README.md                  # Project description
```
