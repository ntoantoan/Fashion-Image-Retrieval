# Image-Caption Training

This repository contains code for training and evaluating an image-caption ranking model. The model uses a combination of image and caption encoders to generate embeddings and a ranker to compute the ranking of candidate images based on their relevance to a given caption.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ntoantoan/Fashion-Image-Retrieval
    cd Fashion-Image-Retrieval
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
To train the model, run the `train.py` script with the desired arguments. For example:

### Training
python train.py --data_set dress --batch_size 3


