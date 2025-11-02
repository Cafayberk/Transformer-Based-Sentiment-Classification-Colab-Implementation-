# Transformer-Based Sentiment Classification (Colab Implementation)

This project implements **sentiment analysis** using a **Transformer Encoder model** built with **PyTorch**.  
It is fully developed and tested in **Google Colab**, featuring a simple dataset of positive and negative sentences.

---

## Features

- Transformer Encoder architecture using PyTorch  
- Custom dataset generation (positive and negative sentences)  
- Training and evaluation pipeline  
- Accuracy and loss monitoring  
- GPU-compatible for faster training in Colab  

---

## Technologies Used

- **Python 3.x**  
- **PyTorch**  
- **NumPy**  
- **scikit-learn**  
- **Google Colab**

---

## Project Structure

Transformer-Sentiment-Classification/
│
├── sentiment_transformer.ipynb # Main Colab notebook
├── dataset_generation.py (optional) # Script for generating sentence dataset
├── README.md # Project documentation
└── requirements.txt (optional)


---


---

## Model Overview

The model uses the **`nn.TransformerEncoder`** architecture to classify text as **positive** or **negative**.  
A linear output layer with a sigmoid activation function is used for binary classification.

Model configuration:
- Embedding Dimension: **128**  
- Hidden Dimension: **64**  
- Number of Heads: **4**  
- Number of Layers: **2**  
- Output: **Binary (Positive / Negative)**  

Example model snippet:
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=num_heads,
    dim_feedforward=hidden_dim
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

## How to Run

Open Google Colab
.

Upload the notebook file:

sentiment_transformer.ipynb


Install the dependencies:

pip install torch numpy scikit-learn


Run all cells sequentially.

Observe the training progress, accuracy, and loss directly in Colab.

## Results

After training, the Transformer model learns to distinguish between positive and negative sentences.
Accuracy and loss graphs can be visualized during and after training.
This project demonstrates the fundamentals of Transformer-based NLP models in a simple, educational setting.

## License

This project is open source and available under the MIT License.
You are free to use, modify, and distribute it for educational or research purposes.

## Author

Ayberk Caf
Developed in Google Colab using PyTorch.
Reinforcement Learning & NLP Enthusiast — 2025
