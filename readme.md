# MultiLingual Text Analyzer

A sophisticated deep learning-based text analysis system that performs 8 simultaneous NLP tasks on text in multiple languages. Built on XLM-RoBERTa, this project provides comprehensive sentiment analysis, emotion detection, intent classification, and safety checks through user-friendly GUI and web API interfaces.

## Features

### Multi-Task Analysis

This analyzer performs 8 different text analysis tasks simultaneously:

* **Sentiment Polarity:** Determines if text is Positive, Negative, or Neutral.
* **Emotion Detection:** Identifies primary emotions (Joy, Anger, Sadness, Surprise, Fear, etc.).
* **Emotional Tone:** Classifies the emotional delivery (Passionate, Calm, Humorous, Pessimistic, etc.).
* **Intent Classification:** Detects the purpose (To Inform, To Persuade, To Complain, To Question, etc.).
* **Formality Analysis:** Categorizes text as Formal, Informal, or Slang.
* **Sarcasm Detection:** Identifies sarcastic content.
* **Toxicity Detection:** Flags toxic or harmful language.
* **Hidden Agenda Detection:** Identifies manipulative or misleading content.

### Multilingual Support

Built on XLM-RoBERTa (Cross-lingual Language Model), supporting analysis in 100+ languages including:

* English
* Spanish
* French
* German
* Hindi
* Tamil
* Chinese
* Arabic
* And many more.

## Architecture

The system uses a Grand Unified Model architecture:

```text
Input Text
    ↓
XLM-RoBERTa Base (Encoder)
    ↓
Dropout Layer (0.3)
    ↓
8 Task-Specific Classification Heads
    ↓
Simultaneous Predictions

```

**Model Specifications:**

* **Base Model:** xlm-roberta-base (279M parameters)
* **Max Sequence Length:** 128 tokens
* **Training Data:** 9,350+ annotated samples
* **Training Framework:** PyTorch with mixed precision (bfloat16)
* **Optimization:** AdamW optimizer with cosine scheduling

## Dataset

The project uses a comprehensive dataset with 10,000+ prompts covering diverse scenarios:

* **Training Set:** 8,547 samples (85%)
* **Test Set:** 1,403 samples (15%)
* **Coverage:** Multiple languages, tones, and communication styles
* **Annotations:** 8 labels per text sample

Sample categories include social media posts, customer reviews, news articles, casual conversations, formal communications, sarcastic remarks, and misleading content.

## Installation

**Prerequisites:**

* Python 3.8 or higher
* CUDA-capable GPU (recommended) or CPU
* 8GB+ RAM (16GB recommended)

**Step 1: Clone the Repository**

```bash
git clone https://github.com/riddhima16/MultiLingual-Text-Analyzer.git
cd MultiLingual-Text-Analyzer

```

**Step 2: Create Virtual Environment**

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt

```

**Step 4: Download Model Files**
Ensure the pre-trained model files are in your repository:

* `9350_model.pth`: Main trained model (1.1GB)
* `Encoders_9350.pkl`: Label encoders
* `10k_prompts.csv`: Training dataset

*(If using Git LFS, ensure files are pulled via `git lfs pull`)*

## Usage

### 1. FastAPI Web Server (Recommended)

Run the interactive web interface and REST API:

```bash
python app.py

```

* Open your browser and navigate to `[http://127.0.0.1:8000](http://127.0.0.1:8000)` to use the modern Tailwind CSS web interface.
* Send programmatic POST requests to `[http://127.0.0.1:8000/analyze](http://127.0.0.1:8000/analyze)` with JSON payload `{"text": "your text here"}`.

### 2. Desktop GUI Application

Run the local Tkinter-based interface for text analysis:

```bash
python main.py

```

1. The application will load the model (may take 10-30 seconds).
2. Enter your text in the input box.
3. Click "Analyze Text" to view comprehensive analysis results in JSON format.

### 3. Training Your Own Model

To train the model from scratch or with your own data:

```bash
python train.py

```

**Training Configuration:**

* Batch Size: 4 (with gradient accumulation of 8)
* Learning Rate: 2e-5
* Epochs: 4
* Optimizer: AdamW with cosine scheduling
* Mixed Precision: Enabled (bfloat16)

*Custom Dataset Format:* Your CSV should include: `text`, `sentiment_polarity`, `emotion`, `emotional_tone`, `intent`, `formality`, `is_sarcastic`, `is_toxic`, `has_hidden_agenda`.

### 4. Model Evaluation

Evaluate model performance with comprehensive visualizations:

```bash
python evaluate_model.py

```

Generates comparative F1-score charts, confusion matrices, performance radar charts, per-class accuracy breakdowns, text length analysis, and model speed comparisons (saved in the `ultimate_evaluation_charts/` directory).

## Model Performance

**Overall Performance (on held-out test set)**

| Task | F1-Score | Accuracy |
| --- | --- | --- |
| Sentiment Polarity | 0.92 | 93% |
| Emotion | 0.88 | 89% |
| Emotional Tone | 0.85 | 86% |
| Intent | 0.87 | 88% |
| Formality | 0.91 | 92% |
| Sarcasm Detection | 0.83 | 85% |
| Toxicity Detection | 0.89 | 91% |
| Hidden Agenda | 0.81 | 84% |

**Inference Speed:**

* CPU: ~15-20 samples/second
* GPU (CUDA): ~80-100 samples/second

## Project Structure

```text
MultiLingual-Text-Analyzer/
│
├── app.py                      # FastAPI web server and frontend UI
├── main.py                     # Desktop GUI application
├── train.py                    # Model training script
├── evaluate_model.py           # Evaluation and visualization script
├── fix_checked.py              # Data processing utility
│
├── 9350_model.pth              # Trained model weights (1.1GB)
├── Encoders_9350.pkl           # Label encoders
├── 10k_prompts.csv             # Training dataset
├── check.csv                   # Test dataset
├── checked.csv                 # Processed test data
│
├── requirements.txt            # Python dependencies
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
│
├── old_504/                    # Legacy experimental models
│   ├── 1_data_generation.py
│   ├── 2_data_processing.py
│   ├── 3_baseline_model_training.py
│   ├── 4_model_enhancement.py
│   ├── 504_model.pth
│   └── final_processed_dataset.csv
│
└── ultimate_evaluation_charts/ # Generated evaluation visualizations

```

## Technical Details

**Key Dependencies:**

* PyTorch 2.3.1 (Deep learning framework)
* Transformers 4.41.2 (Hugging Face model library)
* FastAPI 0.111.0 & Uvicorn 0.29.0 (Web server and API)
* scikit-learn 1.4.2 (ML utilities and metrics)
* pandas 2.2.2 (Data manipulation)
* numpy 1.26.4 (Numerical computing)
* matplotlib 3.8.4 & seaborn 0.13.2 (Visualizations)

**Model Architecture Implementation:**

```python
class GrandUnifiedModel(nn.Module):
    """
    Multi-task learning model with shared encoder and 
    task-specific classification heads.
    """
    def __init__(self, base_model, num_classes_per_task):
        super(GrandUnifiedModel, self).__init__()
        self.base = base_model  # XLM-RoBERTa
        hidden_size = self.base.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.3)
        
        # 8 task-specific heads
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes) 
            for task, num_classes in num_classes_per_task.items()
        })

```

**Training Strategy:**

1. Multi-task Learning: Joint training on all 8 tasks.
2. Class Balancing: Weighted loss for imbalanced classes.
3. Mixed Precision: Automatic mixed precision (AMP) for efficiency.
4. Gradient Accumulation: Effective batch size of 32 (4 × 8).
5. Learning Rate Scheduling: Cosine decay with warmup.

## Use Cases

* **Content Moderation:** Detect toxic, harmful, or misleading content.
* **Customer Service:** Analyze customer sentiment and intent.
* **Social Media Monitoring:** Track brand sentiment and emotions.
* **Market Research:** Understand customer opinions and feedback.
* **Educational Tools:** Analyze writing style and tone.
* **Mental Health:** Detect emotional states in text.
* **Misinformation Detection:** Identify hidden agendas and manipulation.

## Roadmap

* [x] Add REST API endpoint for programmatic access (Completed via FastAPI)
* [ ] Support for batch processing
* [ ] Real-time streaming analysis
* [ ] Fine-tuning on domain-specific data
* [ ] Additional languages and dialects
* [ ] Explainability features (attention visualization)
* [ ] Mobile application

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

*Please ensure code follows PEP 8 style guidelines, tests pass, documentation is updated, and commit messages are descriptive.*

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* **Hugging Face** for the Transformers library and pre-trained models.
* **Facebook AI** for the XLM-RoBERTa architecture.
* **PyTorch Team** for the deep learning framework.
* **The open-source community** for various tools and libraries.

## 👩‍💻 Author

Riddhima Raj 

* GitHub: github.com/riddhima16/MultiLingual-Text-Analyzer

* This project was developed as a case study in high-fidelity synthetic data generation and multi-layered text analysis.
