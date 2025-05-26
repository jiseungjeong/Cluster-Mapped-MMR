import os
import json
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Class for experiment management and result storage"""

    def __init__(
        self, save_dir: str = "./results", experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment manager

        Args:
            save_dir: Directory to save results
            experiment_name: Experiment name (None to use timestamp)
        """
        self.save_dir = save_dir

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"

        # Create results directory
        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        logger.info(
            f"Experiment started: {self.experiment_name} (Save location: {self.experiment_dir})"
        )

        # Results storage
        self.results = []

    def log_result(self, result: Dict[str, Any]):
        """
        Log experiment result

        Args:
            result: Result dictionary
        """
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()

        # Save result
        self.results.append(result)

        # Log output
        dataset = result.get("dataset", "unknown")
        method = result.get("method", "unknown")
        question_id = result.get("question_id", "unknown")
        correct = result.get("correct", None)
        tokens = result.get("tokens", {}).get("total_tokens", 0)

        correct_str = "✓" if correct == True else "✗" if correct == False else "?"
        logger.info(
            f"Result: {dataset}/{method} | Question {question_id} | {correct_str} | Tokens {tokens}"
        )

    def save_results(self, format: str = "both"):
        """
        Save experiment results

        Args:
            format: Save format ('json', 'csv', 'both')
        """
        if not self.results:
            logger.warning("No results to save.")
            return

        # Save JSON
        if format in ["json", "both"]:
            json_path = os.path.join(self.experiment_dir, "results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved: {json_path}")

        # Save CSV
        if format in ["csv", "both"]:
            # Convert to DataFrame (flatten nested structures)
            flat_results = []
            for result in self.results:
                flat_result = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_result[f"{key}_{sub_key}"] = sub_value
                    else:
                        flat_result[key] = value
                flat_results.append(flat_result)

            # Save CSV
            df = pd.DataFrame(flat_results)
            csv_path = os.path.join(self.experiment_dir, "results.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"Results saved: {csv_path}")

    def summarize_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Summarize results

        Returns:
            Summary statistics by method and dataset
        """
        if not self.results:
            logger.warning("No results to summarize.")
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Check basic information
        methods = df["method"].unique()
        datasets = df["dataset"].unique()

        # Summary statistics
        summary = {}

        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            dataset_summary = {}

            for method in methods:
                method_df = dataset_df[dataset_df["method"] == method]

                # Calculate accuracy
                accuracy = (
                    method_df["correct"].mean() if "correct" in method_df else None
                )

                # Token statistics
                tokens = {}
                if "tokens_total_tokens" in method_df.columns:
                    tokens["total"] = {
                        "mean": method_df["tokens_total_tokens"].mean(),
                        "std": method_df["tokens_total_tokens"].std(),
                    }

                if "tokens_prompt_tokens" in method_df.columns:
                    tokens["prompt"] = {
                        "mean": method_df["tokens_prompt_tokens"].mean(),
                        "std": method_df["tokens_prompt_tokens"].std(),
                    }

                if "tokens_completion_tokens" in method_df.columns:
                    tokens["completion"] = {
                        "mean": method_df["tokens_completion_tokens"].mean(),
                        "std": method_df["tokens_completion_tokens"].std(),
                    }

                # Response time
                response_time = {
                    "mean": (
                        method_df["response_time"].mean()
                        if "response_time" in method_df
                        else None
                    ),
                    "std": (
                        method_df["response_time"].std()
                        if "response_time" in method_df
                        else None
                    ),
                }

                # Selection time
                selection_time = {
                    "mean": (
                        method_df["selection_time"].mean()
                        if "selection_time" in method_df
                        else None
                    ),
                    "std": (
                        method_df["selection_time"].std()
                        if "selection_time" in method_df
                        else None
                    ),
                }

                # Total latency
                total_latency = {
                    "mean": (
                        method_df["total_latency"].mean()
                        if "total_latency" in method_df
                        else None
                    ),
                    "std": (
                        method_df["total_latency"].std()
                        if "total_latency" in method_df
                        else None
                    ),
                }

                # Method summary
                method_summary = {
                    "count": len(method_df),
                    "accuracy": accuracy,
                    "tokens": tokens,
                    "response_time": response_time,
                    "selection_time": selection_time,
                    "total_latency": total_latency,
                }

                dataset_summary[method] = method_summary

            summary[dataset] = dataset_summary

        # Save summary
        summary_path = os.path.join(self.experiment_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved: {summary_path}")

        return summary

    def print_summary(self, summary: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Print summary information

        Args:
            summary: Summary information (None to calculate from results)
        """
        if summary is None:
            summary = self.summarize_results()

        if not summary:
            return

        print("\n===== Experiment Results Summary =====")

        for dataset, dataset_summary in summary.items():
            print(f"\nDataset: {dataset}")

            # Table header
            print(
                f"{'Method':<10} {'Accuracy':<10} {'Total Tokens':<12} {'Selection':<10} {'Response':<10} {'Total':<10}"
            )
            print("-" * 70)

            for method, method_summary in dataset_summary.items():
                accuracy = method_summary.get("accuracy")
                accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"

                tokens = method_summary.get("tokens", {}).get("total", {}).get("mean")
                tokens_str = f"{tokens:.1f}" if tokens is not None else "N/A"

                selection = method_summary.get("selection_time", {}).get("mean")
                selection_str = f"{selection:.2f}s" if selection is not None else "N/A"

                response = method_summary.get("response_time", {}).get("mean")
                response_str = f"{response:.2f}s" if response is not None else "N/A"

                total = method_summary.get("total_latency", {}).get("mean")
                total_str = f"{total:.2f}s" if total is not None else "N/A"

                print(
                    f"{method:<10} {accuracy_str:<10} {tokens_str:<12} {selection_str:<10} {response_str:<10} {total_str:<10}"
                )

        print("\n==============================")

    def save_meta(self, meta_data: Dict[str, Any]):
        """
        Save meta information for the experiment

        Args:
            meta_data: Meta data dictionary to save
        """
        meta_path = os.path.join(self.experiment_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Meta data saved: {meta_path}")
