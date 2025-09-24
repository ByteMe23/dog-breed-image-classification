# Dog Breed Image Classification

## About The Project

This repository contains a Python-based application for classifying images of pets, with a special focus on identifying dog breeds. It leverages pre-trained Convolutional Neural Network (CNN) models to classify images and compares the predictions against ground-truth labels derived from the image filenames. The application calculates and presents detailed performance statistics, allowing for a comparative analysis of different CNN architectures.

The primary goal is to evaluate the performance of three popular CNN models—**AlexNet**, **VGG**, and **ResNet**—on the task of dog breed classification.

## Key Features

- **Multi-Model Support**: Classify images using AlexNet, VGG, or ResNet.
- **Label Extraction**: Automatically generates ground-truth labels from image filenames.
- **Performance Evaluation**: Compares model predictions to truth labels and calculates key metrics, including:
    - Percentage of correct classifications.
    - Percentage of correctly identified dog images.
    - Percentage of correctly identified dog breeds.
    - Percentage of correctly identified non-dog images.
- **Dog vs. Not-a-Dog Classification**: Adjusts results to specifically measure the model's ability to distinguish dogs from other animals or objects.
- **Command-Line Interface**: Flexible execution with command-line arguments to specify the image directory, model architecture, and dog names file.
- **Detailed Reporting**: Prints a comprehensive summary of classification results, including misclassified examples.

## Getting Started

Follow these instructions to get a local copy of the project up and running on your machine.

### Prerequisites

This project requires Python 3 and the following libraries:

- PyTorch
- Torchvision
- Pillow

You can install the necessary packages using pip:

```sh
pip install torch torchvision pillow
```

### Usage

The main script for running the classification is `check_images.py`. You can run it from the command line, specifying the directory of images to classify, the CNN model architecture, and the file containing dog names.

**Basic Example:**

This command runs the classification on the `pet_images/` directory using the VGG model.

```sh
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

**Command-Line Arguments:**

- `--dir <directory with images>`: Path to the folder containing the images. (Default: `pet_images/`)
- `--arch <model>`: The CNN model architecture to use. Valid options are `resnet`, `alexnet`, or `vgg`. (Default: `vgg`)
- `--dogfile <file that contains dognames>`: The path to a text file listing dog names, one per line. (Default: `dognames.txt`)

### Running Batch Tests

The repository includes shell scripts to automate the testing of all three models on the provided image sets. The output of each run is saved to a corresponding text file.

- To test on the `pet_images` dataset:
  ```sh
  sh run_models_batch.sh
  ```
- To test on the `uploaded_images` dataset:
  ```sh
  sh run_models_batch_uploaded.sh
  ```

## Project Structure

- `check_images.py`: The main script that orchestrates the entire classification and evaluation process.
- `classifier.py`: Contains the core `classifier` function that loads a pre-trained model and classifies a single image.
- `get_input_args.py`: Parses command-line arguments.
- `get_pet_labels.py`: Creates ground-truth pet labels from the image filenames in the specified directory.
- `classify_images.py`: Iterates through images, calls the `classifier` function, and compares the results to the ground-truth labels.
- `adjust_results4_isadog.py`: Refines the results dictionary by checking if the pet and classifier labels correspond to a dog breed, using `dognames.txt`.
- `calculates_results_stats.py`: Computes various performance statistics based on the classification results.
- `print_results.py`: Displays a formatted summary of the statistics and any misclassifications.
- `dognames.txt`: A list of dog names used to verify if a label refers to a dog.
- `imagenet1000_clsid_to_human.txt`: A dictionary mapping ImageNet class IDs to human-readable labels.
- `pet_images/`: A directory containing the dataset of pet images for classification.
- `uploaded_images/`: A sample directory for classifying user-provided images.
- `run_models_batch.sh` / `run_models_batch_uploaded.sh`: Scripts to run the classifier with all available models.
- `*.txt`: Log files containing the output from model test runs.

## Model Performance Comparison

The shell scripts run all three models on the `pet_images` dataset. Based on the output files, the performance on this dataset is as follows:

| Model   | % Label Match | % Correct Breed | % Correct Not-Dog |
| :------ | :------------ | :-------------- | :---------------- |
| AlexNet | 75.0%         | 80.0%           | 100.0%            |
| ResNet  | 82.5%         | 90.0%           | 90.0%             |
| **VGG** | **87.5%**     | **93.3%**       | **100.0%**        |

The **VGG** architecture demonstrates the highest performance in overall label matching and correct breed identification for the provided `pet_images` dataset.

## Example Output

<img width="1266" height="729" alt="Op" src="https://github.com/user-attachments/assets/40813176-df2e-44fd-9080-00a47c319367" />

```sh
Number of Images: 40
Number of Dog Images: 30
Number of Non-Dog Images: 10
Number of Matches: 36
Correctly Classified Dogs: 28
Correctly Classified Breeds: 25
Percentage Match: 90.0%
Percentage Correct Dogs: 93.3%
Percentage Correct Breeds: 83.3%
Percentage Correct Non-Dogs: 80.0%
```

**Keep learning!!**🚀✨

