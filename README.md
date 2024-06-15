# MultiClass-Classification-Using-PEFT-Adapters
1. Setting up the Environment:

Imports necessary libraries like transformers, datasets, evaluate, and others.
Sets up file paths for data and model storage.
2. Exploring and Understanding Dataset:

Loads the dataset from CSV using HuggingFace datasets library.
Selects relevant columns ("text" and "label") and renames them for clarity.
Prints an example of the text data and label.
3. Data Pre-processing:

Splits the data into training, validation, and test sets (80%/10%/10%).
Creates a small balanced subset of the training and validation data (200 samples per class) for experimentation purposes.
Casts the "label" column to ClassLabel type for better handling of categorical data.
4. Tokenization:

Downloads the tokenizer from the pre-trained Roberta model ("roberta-base").
Defines a function tokenize_fn to tokenize text data in batches using the tokenizer.
Applies the tokenize_fn function to all splits using the map function.
Removes unnecessary columns and renames others for better structure.
Converts the dataset to torch format for compatibility with the model.
5. Model Training:

Loads the Roberta configuration file and modifies it to include id2label and label2id mappings for interpretability.
Creates a RobertaAdapterModel from the pre-trained model and the modified configuration.
Adds a new adapter named "stack_exchange" to the model.
Adds a classification head with 10 labels (one for each class) and the id2label mapping.
Activates the "stack_exchange" adapter for training.
6. Evaluation Metrics:

Defines a function compute_metrics to calculate macro F1 score and accuracy based on model predictions and true labels.
7. Setting up Logger:

Sets up Weights & Biases (W&B) for experiment logging (requires W&B account).
Defines the project name for logging experiments.

8. Check the Best Saved Model
