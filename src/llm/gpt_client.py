import os
import time
import logging
from typing import List, Dict, Any, Optional
import openai
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GPTClient:
    """Client class for calling OpenAI GPT models"""

    def __init__(
        self, model_name: str = None, temperature: float = None, max_tokens: int = None
    ):
        """
        Initialize GPT client

        Args:
            model_name: GPT model name to use
            temperature: Generation diversity parameter (0~1)
            max_tokens: Maximum number of tokens to generate
        """
        # Set API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

        # Model settings
        self.model_name = model_name or os.getenv("GPT_MODEL", "gpt-3.5-turbo")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("GPT_TEMPERATURE", "0.0"))
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("GPT_MAX_TOKENS", "1024"))
        )

        logger.info(
            f"GPT client initialized: model={self.model_name}, temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def create_cot_prompt(
        self, question: str, examples: List[Dict], cot_format: bool = True
    ) -> str:
        """
        Create few-shot CoT prompt

        Args:
            question: Current question
            examples: List of examples, each example should have 'question' and 'answer' keys
            cot_format: Whether to use Chain-of-Thought format

        Returns:
            Generated prompt string
        """
        # Set system message
        if cot_format:
            system_message = (
                "Solve the following problem step-by-step. Explain your thinking at each step, "
                "and provide the final answer. At the end, write your final answer as a number or letter after '####'."
            )
        else:
            system_message = "Solve the following problem."

        prompt = f"{system_message}\n\n"

        # Add examples
        for i, example in enumerate(examples):
            prompt += f"Question: {example['question']}\n"

            if cot_format and "reasoning" in example and example["reasoning"]:
                prompt += f"Reasoning: {example['reasoning']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
            else:
                prompt += f"Answer: {example['full_answer'] if 'full_answer' in example else example['answer']}\n\n"

        # Add current question
        prompt += f"Question: {question}\n"
        prompt += "Reasoning:"

        return prompt

    def call_gpt(
        self, prompt: str, retries: int = 3, backoff_factor: float = 2.0
    ) -> Dict[str, Any]:
        """
        Call GPT model

        Args:
            prompt: Input prompt
            retries: Number of retry attempts
            backoff_factor: Retry wait time increase factor

        Returns:
            GPT response results and metadata
        """
        attempt = 0
        last_error = None
        wait_time = 1.0

        start_time = time.time()

        while attempt < retries:
            try:
                # Call OpenAI API
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                end_time = time.time()

                # Process response
                result = {
                    "response": response.choices[0].message.content.strip(),
                    "tokens": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "model": response.model,
                    "response_time": end_time - start_time,
                }

                return result

            except Exception as e:
                attempt += 1
                last_error = str(e)

                if attempt < retries:
                    logger.warning(
                        f"GPT API call failed (attempt {attempt}/{retries}): {last_error}. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    wait_time *= backoff_factor
                else:
                    logger.error(
                        f"GPT API call failed after {retries} attempts: {last_error}"
                    )

        # All retry attempts failed
        raise Exception(f"GPT API call failed: {last_error}")

    def extract_answer(self, response: str) -> str:
        """
        Extract final answer from GPT response

        Args:
            response: GPT response string

        Returns:
            Extracted answer
        """
        # Consider content after #### as the answer
        parts = response.split("####")
        if len(parts) > 1:
            return parts[1].strip()

        # Consider the last line as the answer
        lines = response.strip().split("\n")
        if lines:
            return lines[-1].strip()

        return response.strip()
