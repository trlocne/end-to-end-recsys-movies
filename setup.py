from setuptools import setup, find_packages

setup(
    name="end-to-end-recommendation-system",
    version="0.1.0",
    description="End-to-end recommendation system with feature store, ML models, and real-time serving",
    author="RecSys Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tritonclient[grpc]>=2.40.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "feast>=0.31.0",
        "kafka-python>=2.0.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "mlflow": [
            "mlflow>=2.0.0",
        ],
        "flink": [
            "apache-flink>=1.18.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
