# Entries Extraction Module

This module provides functionality to extract and group semantically related sentences from text documents using SetFit models. It's particularly useful for processing humanitarian situation reports and similar documents where information needs to be chunked into coherent entries.

## Features

- Extracts relevant sentences using a pre-trained SetFit model
- Groups related sentences together based on semantic independence
- Handles long text passages by chunking them into manageable sizes
- Supports customizable maximum sentence length and overlap parameters

## Installation

To use this module, you can install it via pip:

```bash
pip install setfit nltk tqdm
```

## Models

The module uses two SetFit models:
1. Sentence Relevancy Model: Determines if a sentence contains relevant information
2. Sentence Independence Model: Determines if sentences should be grouped together

## Usage

```python
from setfit_extraction import EntriesExtractor

# Initialize the extractor
extractor = EntriesExtractor(
    relevancy_model_name="Sfekih/sentence_relevancy_model",
    independance_model_name="Sfekih/sentence_independancy_model",
    max_sentences=5,
    overlap=2
)

# Process a single document or multiple documents
documents = [
    "Your document text here. This contains multiple sentences. Some sentences are relevant.",
    "Another document with different content. More sentences here."
]

# Extract entries
entries = extractor(documents)
```

## Configuration Parameters

The `EntriesExtractor` class accepts the following parameters:

- `relevancy_model_name` (str): Path or name of the SetFit model for determining sentence relevancy
- `independance_model_name` (str): Path or name of the SetFit model for determining sentence independence
- `max_sentences` (int): Maximum number of sentences to group together (default: 5)
- `overlap` (int): Number of sentences to overlap when chunking long passages (default: 2)

## Output Format

The extractor returns a nested list structure:
- Top level: List of documents
- Second level: List of entries for each document
- Third level: List of sentences forming a coherent entry

Example output:
```python
[
    # Document 1
    [
        ["Sentence 1", "Sentence 2"],  # Entry 1
        ["Sentence 3", "Sentence 4"],  # Entry 2
    ],
    # Document 2
    [
        ["Sentence 1"],                # Entry 1
        ["Sentence 2", "Sentence 3"],  # Entry 2
    ]
]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

## Next Steps

- Add training data and code.
- Train multilingual models.