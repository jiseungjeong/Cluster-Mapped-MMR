from setuptools import setup, find_packages

setup(
    name="cot-example-selection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "openai>=1.6.0",
        "matplotlib>=3.7.0",
        "hdbscan>=0.8.30",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    author="CoT 연구팀",
    author_email="example@example.com",
    description="클러스터링 기반 CoT 예제 선택 기법 실험",
    keywords="CoT, Chain-of-Thought, clustering, mmr, example selection",
    python_requires=">=3.8",
)
