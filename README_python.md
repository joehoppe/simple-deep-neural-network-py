# Neural Network (Python)

A port of the [TypeScript Simple Deep Neural Network](https://github.com/joehoppe?tab=repositories) to Python

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.perceptron.services.perceptron_service import PerceptronService
from src.activation.services.binary_step_service import BinaryStepService

activation = BinaryStepService(0.5)
perceptron = PerceptronService(activation)
perceptron.set_weights([0.8, 0.4])
result = perceptron.predict([1, 0])
```

## Development

```bash
python -m pytest                    # Run tests
python -m pytest --cov=src          # Run tests with coverage
flake8 src/                         # Lint code
black src/                          # Format code
```

## Running

```bash
python main.py
```
