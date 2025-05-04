# COMETHINT

ComeTHint (COMET Internationale Thai) is a specialized adaptation of the COMET framework optimized for Thai language machine translation evaluation. Built on top of the powerful COMET metrics, ComeTHint features fine-tuning with Thai language data and various optimization techniques to enhance performance for Thai-specific translation evaluation.

## Quick Installation

ComeTHint requires Python 3.8 or above. Simple installation from PyPI:

```bash
pip install --upgrade pip
pip install cometh
```

**Note:** To use some ComeTHint models, you must acknowledge its license on Hugging Face Hub and log in to Hugging Face Hub.

For local development:

```bash
git clone https://github.com/COMETH-TEAM/COMETHINT
cd COMETHINT
pip install poetry
poetry install
```

## Features

- **Thai Optimization**: Specifically fine-tuned with Thai language data
- **Multi-adaptation Techniques**: Implements DoRA (Dense-or-Rank Adaptation) and other efficient parameter adaptation methods
- **Error Classification**: Identifies and classifies translation errors with Thai-specific error categories
- **Context-aware Evaluation**: Supports document-level context for better discourse evaluation

## Using ComeTHint in Python

```python
from comethint import download_model, load_from_checkpoint

# Choose your model
model_path = download_model("comet/...")

# Load the model
model = load_from_checkpoint(model_path)

# Prepare data
data = [
    {
        "src": "ฉันต้องการสั่งอาหาร ส่งถึงใน 10-15 นาทีได้ไหม",
        "mt": "I would like to order food. Can it be delivered in 10-15 minutes?",
        "ref": "I want to order food. Can it arrive in 10-15 minutes?"
    }
]

# Get predictions
model_output = model.predict(data, batch_size=8, gpus=1)
```

## Thai Language Support

ComeTHint builds upon the multilingual foundation of XLM-R with additional fine-tuning specifically for Thai language structures, including:

- Better handling of Thai script and character boundaries
- Improved understanding of Thai linguistic patterns
- Enhanced recognition of Thai-specific translation challenges

## Training Your Own Metric

```bash
comethint-train --cfg configs/models/{your_model_config}.yaml
```

You can then use your own metric:

```bash
comethint-score -s src.th -t hyp1.en -r ref.en --model PATH/TO/CHECKPOINT
```

## Testing

```bash
poetry run coverage run --source=comethint -m unittest discover
poetry run coverage report -m
```

## Publications

If you use ComeTHint in your work, please cite:

- Original COMET paper: [COMET: A Neural Framework for MT Evaluation](https://www.aclweb.org/anthology/2020.emnlp-main.213)
- ComeTHint adaptation techniques (upcoming publication)

## License

ComeTHint is released under the Apache License 2.0.