import os
import json
import pandas as pd
import requests
import zipfile
import io
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
import re

logger = logging.getLogger(__name__)


class Dataset:
    """Dataset class: Loading and managing datasets"""

    def __init__(self, name: str, data_dir: str = "./data"):
        """
        Initialize dataset

        Args:
            name: Dataset name ('gsm8k', 'commonsenseqa', 'arc', or 'combined')
            data_dir: Data directory path
        """
        self.name = name.lower()
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.examples_dir = os.path.join(data_dir, "examples")

        # Check directories
        for directory in [self.raw_dir, self.processed_dir, self.examples_dir]:
            os.makedirs(directory, exist_ok=True)

        self.test_data = None
        self.example_pool = None

        # Load dataset
        self._load_or_prepare_data()

    def _load_or_prepare_data(self):
        """Load or prepare dataset"""
        # Combined dataset handling
        if self.name == "combined":
            self._load_combined_data()
            return

        # Check for enhanced version for ARC
        if self.name == "arc":
            test_enhanced_path = os.path.join(
                self.processed_dir, f"{self.name}_test_enhanced.json"
            )
            examples_enhanced_path = os.path.join(
                self.examples_dir, f"{self.name}_examples_enhanced.json"
            )

            if os.path.exists(test_enhanced_path) and os.path.exists(
                examples_enhanced_path
            ):
                logger.info(f"Loading enhanced version: {test_enhanced_path}")
                with open(test_enhanced_path, "r", encoding="utf-8") as f:
                    self.test_data = json.load(f)

                logger.info(f"Loading enhanced version: {examples_enhanced_path}")
                with open(examples_enhanced_path, "r", encoding="utf-8") as f:
                    self.example_pool = json.load(f)

                logger.info(
                    f"Enhanced ARC dataset loaded: {len(self.test_data)} test items, {len(self.example_pool)} examples"
                )
                return

        # Check for enhanced version for CommonsenseQA
        if self.name == "commonsenseqa":
            test_enhanced_path = os.path.join(
                self.processed_dir, f"{self.name}_test_enhanced.json"
            )
            examples_enhanced_path = os.path.join(
                self.examples_dir, f"{self.name}_examples_enhanced.json"
            )

            if os.path.exists(test_enhanced_path) and os.path.exists(
                examples_enhanced_path
            ):
                logger.info(f"Loading enhanced version: {test_enhanced_path}")
                with open(test_enhanced_path, "r", encoding="utf-8") as f:
                    self.test_data = json.load(f)

                logger.info(f"Loading enhanced version: {examples_enhanced_path}")
                with open(examples_enhanced_path, "r", encoding="utf-8") as f:
                    self.example_pool = json.load(f)

                logger.info(
                    f"Enhanced dataset loaded: {len(self.test_data)} test items, {len(self.example_pool)} examples"
                )
                return

        # Normal dataset loading
        test_path = os.path.join(self.processed_dir, f"{self.name}_test.json")
        examples_path = os.path.join(self.examples_dir, f"{self.name}_examples.json")

        # Check if processed data exists
        if os.path.exists(test_path) and os.path.exists(examples_path):
            logger.info(f"Loading: {test_path}")
            with open(test_path, "r", encoding="utf-8") as f:
                self.test_data = json.load(f)

            logger.info(f"Loading: {examples_path}")
            with open(examples_path, "r", encoding="utf-8") as f:
                self.example_pool = json.load(f)
        else:
            logger.info(
                f"Could not find processed data. Preparing dataset {self.name}..."
            )
            self._prepare_dataset()

    def _load_combined_data(self):
        """Load combined dataset files"""
        test_path = os.path.join(self.processed_dir, "combined_test.json")
        examples_path = os.path.join(self.examples_dir, "combined_examples.json")

        if not os.path.exists(test_path):
            logger.error(f"Combined test file not found: {test_path}")
            raise FileNotFoundError(f"Combined test file not found: {test_path}")

        if not os.path.exists(examples_path):
            logger.error(f"Combined examples file not found: {examples_path}")
            raise FileNotFoundError(
                f"Combined examples file not found: {examples_path}"
            )

        logger.info(f"Loading combined test data: {test_path}")
        with open(test_path, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)

        logger.info(f"Loading combined examples: {examples_path}")
        with open(examples_path, "r", encoding="utf-8") as f:
            self.example_pool = json.load(f)

        logger.info(
            f"Loaded combined dataset: {len(self.test_data)} test items, {len(self.example_pool)} examples"
        )

    def _prepare_dataset(self):
        """Prepare and process dataset"""
        if self.name == "gsm8k":
            self._prepare_gsm8k()
        elif self.name == "commonsenseqa":
            self._prepare_commonsenseqa()
        elif self.name == "arc":
            self._prepare_arc()
        elif self.name == "combined":
            self._load_combined_data()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

    def _download_file(self, url: str, save_path: str) -> bool:
        """Download file"""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception if error occurs

            with open(save_path, "wb") as f:
                f.write(response.content)

            logger.info(f"File download complete: {save_path}")
            return True
        except Exception as e:
            logger.error(f"File download failed: {url}, Error: {str(e)}")
            return False

    def _download_and_extract_zip(self, url: str, extract_dir: str) -> bool:
        """Download and extract ZIP file"""
        try:
            response = requests.get(url)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"ZIP file download and extraction complete: {extract_dir}")
            return True
        except Exception as e:
            logger.error(
                f"ZIP file download or extraction failed: {url}, Error: {str(e)}"
            )
            return False

    def _prepare_gsm8k(self):
        """Prepare GSM8K dataset"""
        # GSM8K dataset URL
        train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

        # Local file path
        train_path = os.path.join(self.raw_dir, "gsm8k_train.jsonl")
        test_path = os.path.join(self.raw_dir, "gsm8k_test.jsonl")

        # Download data
        if not os.path.exists(train_path):
            self._download_file(train_url, train_path)

        if not os.path.exists(test_path):
            self._download_file(test_url, test_path)

        # Load data
        train_data = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))

        test_data = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                test_data.append(json.loads(line))

        logger.info(
            f"GSM8K dataset loaded: Training {len(train_data)} items, Test {len(test_data)} items"
        )

        # Process data
        processed_train = []
        for i, item in enumerate(train_data):
            # Extract answer
            answer_parts = item["answer"].split("####")
            reasoning = (
                answer_parts[0].strip() if len(answer_parts) > 1 else item["answer"]
            )
            answer = answer_parts[1].strip() if len(answer_parts) > 1 else ""

            processed_train.append(
                {
                    "id": f"gsm8k_train_{i}",
                    "question": item["question"],
                    "reasoning": reasoning,
                    "answer": answer,
                    "full_answer": item["answer"],
                }
            )

        processed_test = []
        for i, item in enumerate(test_data):
            # Extract answer
            answer_parts = item["answer"].split("####")
            reasoning = (
                answer_parts[0].strip() if len(answer_parts) > 1 else item["answer"]
            )
            answer = answer_parts[1].strip() if len(answer_parts) > 1 else ""

            processed_test.append(
                {
                    "id": f"gsm8k_test_{i}",
                    "question": item["question"],
                    "reasoning": reasoning,
                    "answer": answer,
                    "full_answer": item["answer"],
                }
            )

        # Construct example pool (select from training data)
        random.seed(42)  # Set seed for reproducibility
        if len(processed_train) > 500:
            example_pool = random.sample(processed_train, 500)
        else:
            example_pool = processed_train.copy()

        # Sample test data (select from test data)
        if len(processed_test) > 100:
            test_samples = random.sample(processed_test, 100)
        else:
            test_samples = processed_test.copy()

        # Save data
        self._save_processed_data(test_samples, example_pool)

        self.test_data = test_samples
        self.example_pool = example_pool

    def _prepare_commonsenseqa(self):
        """Prepare CommonsenseQA dataset"""
        # CommonsenseQA dataset URL
        train_url = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
        dev_url = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
        test_url = (
            "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"
        )

        # Local file path
        train_path = os.path.join(self.raw_dir, "commonsenseqa_train.jsonl")
        dev_path = os.path.join(self.raw_dir, "commonsenseqa_dev.jsonl")
        test_path = os.path.join(self.raw_dir, "commonsenseqa_test.jsonl")

        # Download data
        if not os.path.exists(train_path):
            self._download_file(train_url, train_path)

        if not os.path.exists(dev_path):
            self._download_file(dev_url, dev_path)

        if not os.path.exists(test_path):
            self._download_file(test_url, test_path)

        # Load data
        train_data = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))

        dev_data = []
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                dev_data.append(json.loads(line))

        logger.info(
            f"CommonsenseQA dataset loaded: Training {len(train_data)} items, Development {len(dev_data)} items"
        )

        # Process data
        processed_train = []
        for i, item in enumerate(train_data):
            # Construct question and choices
            question = item["question"]["stem"]
            choices = item["question"]["choices"]
            choices_text = []

            for choice in choices:
                choices_text.append(f"{choice['label']}. {choice['text']}")

            choices_str = "\n".join(choices_text)
            formatted_question = f"{question}\n{choices_str}"

            # Find answer
            answer_key = item["answerKey"]
            answer_text = next(
                (c["text"] for c in choices if c["label"] == answer_key), ""
            )

            # Generate reasoning (since actual data does not have reasoning, generate dummy reasoning)
            reasoning = (
                f"The most common sense answer to this question is '{answer_text}'."
            )

            processed_train.append(
                {
                    "id": f"commonsenseqa_train_{i}",
                    "question": formatted_question,
                    "reasoning": reasoning,
                    "answer": f"{answer_key}. {answer_text}",
                    "answerKey": answer_key,
                }
            )

        processed_dev = []
        for i, item in enumerate(dev_data):
            # Construct question and choices
            question = item["question"]["stem"]
            choices = item["question"]["choices"]
            choices_text = []

            for choice in choices:
                choices_text.append(f"{choice['label']}. {choice['text']}")

            choices_str = "\n".join(choices_text)
            formatted_question = f"{question}\n{choices_str}"

            # Find answer
            answer_key = item["answerKey"]
            answer_text = next(
                (c["text"] for c in choices if c["label"] == answer_key), ""
            )

            # Generate reasoning (since actual data does not have reasoning, generate dummy reasoning)
            reasoning = (
                f"The most common sense answer to this question is '{answer_text}'."
            )

            processed_dev.append(
                {
                    "id": f"commonsenseqa_dev_{i}",
                    "question": formatted_question,
                    "reasoning": reasoning,
                    "answer": f"{answer_key}. {answer_text}",
                    "answerKey": answer_key,
                }
            )

        # Construct example pool (select from training data)
        random.seed(42)  # Set seed for reproducibility
        if len(processed_train) > 500:
            example_pool = random.sample(processed_train, 500)
        else:
            example_pool = processed_train.copy()

        # Sample test data (select from development data)
        if len(processed_dev) > 100:
            test_samples = random.sample(processed_dev, 100)
        else:
            test_samples = processed_dev.copy()

        # Save data
        self._save_processed_data(test_samples, example_pool)

        self.test_data = test_samples
        self.example_pool = example_pool

    def _prepare_arc(self):
        """Prepare ARC dataset"""
        # ARC dataset is downloaded separately
        # This method would be called if arc_test.json and arc_examples.json don't exist
        # but arc_test_enhanced.json and arc_examples_enhanced.json are expected to be provided by the user

        logger.error("ARC dataset requires manually prepared enhanced files")
        raise FileNotFoundError(
            f"ARC dataset files not found. Please provide {self.name}_test.json and {self.name}_examples.json files"
            f" or enhanced versions in the appropriate directories."
        )

    def _save_processed_data(self, test_data, example_pool):
        """Save processed data"""
        test_path = os.path.join(self.processed_dir, f"{self.name}_test.json")
        examples_path = os.path.join(self.examples_dir, f"{self.name}_examples.json")

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(example_pool, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed data saved: {test_path}, {examples_path}")

    def get_test_data(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get test dataset

        Args:
            n: Number of data to get (None for all)

        Returns:
            List of test data
        """
        if n is not None:
            return self.test_data[: min(n, len(self.test_data))]
        return self.test_data

    def get_example_pool(self, n: Optional[int] = None) -> List[Dict]:
        """
        Get example pool

        Args:
            n: Number of examples to get (None for all)

        Returns:
            List of example pool
        """
        if n is not None:
            return self.example_pool[: min(n, len(self.example_pool))]
        return self.example_pool
