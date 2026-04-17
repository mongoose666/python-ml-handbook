---
title: "The Python ML & AI Libraries Handbook"
subtitle: "A Complete Reference Guide to the Python Machine Learning & Artificial Intelligence Ecosystem"
author: "mongoose666"
date: "April 2026"
---

# The Python ML & AI Libraries Handbook

A practical, publication-ready reference for the modern Python ML/AI ecosystem.

---

## Table of Contents

- [1. 🧱 Foundational Numerical & Scientific Libraries](#sec1)
  - [NumPy](#numpy)
  - [SciPy](#scipy)
  - [Numba](#numba)
  - [CuPy](#cupy)
- [2. 🤖 Traditional Machine Learning](#sec2)
  - [scikit-learn](#scikit-learn)
  - [XGBoost](#xgboost)
  - [LightGBM](#lightgbm)
  - [CatBoost](#catboost)
  - [StatsModels](#statsmodels)
- [3. 🔥 Deep Learning Frameworks](#sec3)
  - [PyTorch](#pytorch)
  - [TensorFlow / Keras](#tensorflow-keras)
  - [JAX](#jax)
  - [MindSpore](#mindspore)
  - [MXNet](#mxnet)
- [4. 📦 Data Handling & Pipelines](#sec4)
  - [Pandas](#pandas)
  - [Polars](#polars)
  - [Dask](#dask)
  - [PyArrow](#pyarrow)
  - [HuggingFace Datasets](#huggingface-datasets)
- [5. 🧠 NLP, Transformers & LLM Ecosystem](#sec5)
  - [HuggingFace Transformers](#huggingface-transformers)
  - [SentenceTransformers](#sentencetransformers)
  - [spaCy](#spacy)
  - [NLTK](#nltk)
  - [OpenAI Python SDK](#openai-python-sdk)
  - [LangChain](#langchain)
  - [LlamaIndex](#llamaindex)
- [6. 👁️ Computer Vision](#sec6)
  - [OpenCV](#opencv)
  - [TorchVision](#torchvision)
  - [Detectron2](#detectron2)
  - [MMDetection / MMEngine](#mmdetection-mmengine)
  - [mediapipe](#mediapipe)
- [7. 📊 Visualization](#sec7)
  - [Matplotlib](#matplotlib)
  - [Seaborn](#seaborn)
  - [Plotly](#plotly)
  - [TensorBoard](#tensorboard)
  - [Weights & Biases (wandb)](#weights--biases-wandb)
- [8. 🧬 Specialized AI Domains](#sec8)
  - [Time Series](#time-series)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Audio / Speech](#audio--speech)
  - [Graph ML](#graph-ml)
- [9. 🧩 Core Essentials Stack](#sec9)
  - [Minimum Viable Stack](#minimum-viable-stack)
  - [Day 1 Script](#day-1-script)
- [10. 📚 Full Resource Reference](#sec10)

---

<a id="sec1"></a>
## 1. 🧱 Foundational Numerical & Scientific Libraries

<a id="numpy"></a>
### NumPy

- Docs: https://numpy.org/doc/
- GitHub: https://github.com/numpy/numpy
- PyPI: https://pypi.org/project/numpy/

```python
import numpy as np

# Create arrays and matrices
x = np.array([1.0, 2.0, 3.0])
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[2.0, 0.0], [1.0, 2.0]])

# Broadcasting: add vector to each matrix row
broadcasted = A + np.array([10.0, 20.0])

# Linear algebra: matrix multiplication and eigenvalues
C = A @ B
eigenvalues, eigenvectors = np.linalg.eig(A)

print(broadcasted)
print(C)
print(eigenvalues)
```

<a id="scipy"></a>
### SciPy

- Docs: https://docs.scipy.org/doc/scipy/
- GitHub: https://github.com/scipy/scipy
- PyPI: https://pypi.org/project/scipy/

```python
import numpy as np
from scipy.optimize import minimize
from scipy import signal

# Optimization: find minimum of a convex function
f = lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2
result = minimize(f, x0=np.array([0.0, 0.0]))

# Signal processing: smooth a noisy sine wave with Savitzky-Golay filter
t = np.linspace(0, 1, 200)
raw = np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(t.size)
smoothed = signal.savgol_filter(raw, window_length=21, polyorder=3)

print(result.x, result.fun)
print(smoothed[:5])
```

<a id="numba"></a>
### Numba

- Docs: https://numba.readthedocs.io/
- GitHub: https://github.com/numba/numba
- PyPI: https://pypi.org/project/numba/

```python
from numba import jit

# JIT compile a loop-heavy function
@jit(nopython=True)
def sum_of_squares(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += i * i
    return total

print(sum_of_squares(10_000_000))
```

<a id="cupy"></a>
### CuPy

- Docs: https://docs.cupy.dev/
- GitHub: https://github.com/cupy/cupy
- PyPI: https://pypi.org/project/cupy-cuda12x/

```python
import cupy as cp

# Allocate arrays on GPU and run vectorized ops
x = cp.arange(1_000_000, dtype=cp.float32)
y = cp.sin(x) + cp.cos(x)

# GPU linear algebra
A = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
B = cp.array([[2.0, 0.0], [1.0, 2.0]], dtype=cp.float32)
C = A @ B

print(cp.asnumpy(C))
```

---

<a id="sec2"></a>
## 2. 🤖 Traditional Machine Learning

<a id="scikit-learn"></a>
### scikit-learn

- Docs: https://scikit-learn.org/stable/
- GitHub: https://github.com/scikit-learn/scikit-learn
- PyPI: https://pypi.org/project/scikit-learn/

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Toy tabular dataset
X = pd.DataFrame(
    {
        "age": [25, 45, 31, 52, 23, 40, 36, 28],
        "income": [40_000, 80_000, 55_000, 95_000, 35_000, 70_000, 62_000, 48_000],
        "city": ["A", "B", "A", "C", "A", "B", "C", "B"],
    }
)
y = [0, 1, 0, 1, 0, 1, 1, 0]

num_features = ["age", "income"]
cat_features = ["city"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
            num_features,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

<a id="xgboost"></a>
### XGBoost

- Docs: https://xgboost.readthedocs.io/
- GitHub: https://github.com/dmlc/xgboost
- PyPI: https://pypi.org/project/xgboost/

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Train/test split on a built-in dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, proba))
```

<a id="lightgbm"></a>
### LightGBM

- Docs: https://lightgbm.readthedocs.io/
- GitHub: https://github.com/microsoft/LightGBM
- PyPI: https://pypi.org/project/lightgbm/

```python
from lightgbm import LGBMClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Multi-class classification with LightGBM
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
```

<a id="catboost"></a>
### CatBoost

- Docs: https://catboost.ai/en/docs/
- GitHub: https://github.com/catboost/catboost
- PyPI: https://pypi.org/project/catboost/

```python
import pandas as pd
from catboost import CatBoostClassifier

# CatBoost handles categorical features directly
X = pd.DataFrame(
    {
        "color": ["red", "blue", "red", "green", "blue", "green"],
        "size": ["S", "M", "L", "M", "S", "L"],
        "weight": [1.2, 2.3, 2.8, 2.1, 1.1, 3.0],
    }
)
y = [0, 1, 1, 1, 0, 1]

cat_features = [0, 1]  # indices of categorical columns
model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=False)
model.fit(X, y, cat_features=cat_features)

print(model.predict(X))
```

<a id="statsmodels"></a>
### StatsModels

- Docs: https://www.statsmodels.org/stable/index.html
- GitHub: https://github.com/statsmodels/statsmodels
- PyPI: https://pypi.org/project/statsmodels/

```python
import numpy as np
import statsmodels.api as sm

# OLS regression and statistical summary
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
beta = np.array([1.5, -2.0])
y = 3.0 + X @ beta + rng.normal(scale=0.5, size=200)

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())
```

---

<a id="sec3"></a>
## 3. 🔥 Deep Learning Frameworks

<a id="pytorch"></a>
### PyTorch

- Docs: https://pytorch.org/docs/stable/index.html
- GitHub: https://github.com/pytorch/pytorch
- PyPI: https://pypi.org/project/torch/

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple CNN for image classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy mini training loop
x_batch = torch.randn(32, 1, 28, 28)
y_batch = torch.randint(0, 10, (32,))

for _ in range(3):
    optimizer.zero_grad()
    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()

print(float(loss))
```

<a id="tensorflow-keras"></a>
### TensorFlow / Keras

- TensorFlow Docs: https://www.tensorflow.org/api_docs
- TensorFlow GitHub: https://github.com/tensorflow/tensorflow
- TensorFlow PyPI: https://pypi.org/project/tensorflow/
- Keras Docs: https://keras.io/
- Keras GitHub: https://github.com/keras-team/keras
- Keras PyPI: https://pypi.org/project/keras/

```python
import numpy as np
from tensorflow import keras

# Build and train a small Sequential model
model = keras.Sequential(
    [
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

X = np.random.randn(256, 20)
y = np.random.randint(0, 2, size=(256, 1))
model.fit(X, y, epochs=3, batch_size=32, verbose=0)

print(model.evaluate(X, y, verbose=0))
```

<a id="jax"></a>
### JAX

- Docs: https://jax.readthedocs.io/
- GitHub: https://github.com/jax-ml/jax
- PyPI: https://pypi.org/project/jax/

```python
import jax
import jax.numpy as jnp

# grad: derivative of scalar function
f = lambda x: jnp.sum(x ** 2)
grad_f = jax.grad(f)

# jit: compile function for speed
@jax.jit
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

# vmap: vectorize scalar function over batch
square = lambda x: x * x
batched_square = jax.vmap(square)

x = jnp.array([1.0, 2.0, 3.0])
print(grad_f(x))
print(normalize(x))
print(batched_square(jnp.array([1.0, 2.0, 3.0, 4.0])))
```

<a id="mindspore"></a>
### MindSpore

MindSpore is Huawei's deep learning framework for training and inference across cloud, edge, and devices.

- Docs: https://www.mindspore.cn/en
- GitHub: https://github.com/mindspore-ai/mindspore
- PyPI: https://pypi.org/project/mindspore/

<a id="mxnet"></a>
### MXNet

Apache MXNet is a mature deep learning framework; it is less common today than PyTorch/TensorFlow but still relevant in legacy stacks.

- Docs: https://mxnet.apache.org/
- GitHub: https://github.com/apache/mxnet
- PyPI: https://pypi.org/project/mxnet/

---

<a id="sec4"></a>
## 4. 📦 Data Handling & Pipelines

<a id="pandas"></a>
### Pandas

- Docs: https://pandas.pydata.org/docs/
- GitHub: https://github.com/pandas-dev/pandas
- PyPI: https://pypi.org/project/pandas/

```python
import pandas as pd

# Read CSV, filter, and aggregate
# df = pd.read_csv("sales.csv")

df = pd.DataFrame(
    {
        "region": ["EU", "EU", "US", "US", "APAC"],
        "sales": [100, 150, 120, 170, 130],
        "product": ["A", "B", "A", "B", "A"],
    }
)

filtered = df[df["sales"] > 120]
summary = df.groupby("region", as_index=False)["sales"].mean()

print(filtered)
print(summary)
```

<a id="polars"></a>
### Polars

- Docs: https://docs.pola.rs/
- GitHub: https://github.com/pola-rs/polars
- PyPI: https://pypi.org/project/polars/

```python
import polars as pl

# Lazy query plan for scalable dataframe transforms
lf = pl.DataFrame(
    {
        "city": ["A", "B", "A", "C"],
        "value": [10, 25, 18, 7],
    }
).lazy()

result = (
    lf.filter(pl.col("value") > 10)
    .group_by("city")
    .agg(pl.col("value").mean().alias("avg_value"))
    .collect()
)

print(result)
```

<a id="dask"></a>
### Dask

- Docs: https://docs.dask.org/
- GitHub: https://github.com/dask/dask
- PyPI: https://pypi.org/project/dask/

```python
import dask.dataframe as dd
import pandas as pd

# Parallel dataframe operations across partitions
pdf = pd.DataFrame({"x": range(1000), "y": range(1000)})
ddf = dd.from_pandas(pdf, npartitions=8)

result = ddf[ddf["x"] % 2 == 0]["y"].mean().compute()
print(result)
```

<a id="pyarrow"></a>
### PyArrow

- Docs: https://arrow.apache.org/docs/python/
- GitHub: https://github.com/apache/arrow
- PyPI: https://pypi.org/project/pyarrow/

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Parquet write/read with Arrow table
table = pa.table({"id": [1, 2, 3], "score": [0.8, 0.9, 0.95]})
pq.write_table(table, "example.parquet")
loaded = pq.read_table("example.parquet")

print(loaded.to_pandas())
```

<a id="huggingface-datasets"></a>
### HuggingFace Datasets

- Docs: https://huggingface.co/docs/datasets
- GitHub: https://github.com/huggingface/datasets
- PyPI: https://pypi.org/project/datasets/

```python
from datasets import load_dataset

# Load a public dataset from the Hub
dataset = load_dataset("ag_news", split="train[:1%]")
print(dataset[0])
```

---

<a id="sec5"></a>
## 5. 🧠 NLP, Transformers & LLM Ecosystem

<a id="huggingface-transformers"></a>
### HuggingFace Transformers

- Docs: https://huggingface.co/docs/transformers
- GitHub: https://github.com/huggingface/transformers
- PyPI: https://pypi.org/project/transformers/

```python
from transformers import pipeline

# Sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("This handbook is incredibly useful.")
print(result)
```

<a id="sentencetransformers"></a>
### SentenceTransformers

- Docs: https://www.sbert.net/
- GitHub: https://github.com/UKPLab/sentence-transformers
- PyPI: https://pypi.org/project/sentence-transformers/

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Encode text and compare semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = ["Machine learning is fun", "I enjoy training models"]
embeddings = model.encode(sentences)
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

print(float(sim))
```

<a id="spacy"></a>
### spaCy

- Docs: https://spacy.io/usage
- GitHub: https://github.com/explosion/spaCy
- PyPI: https://pypi.org/project/spacy/

```python
import spacy

# Named Entity Recognition (requires a downloaded model)
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
doc = nlp("OpenAI released a model in San Francisco.")

for ent in doc.ents:
    print(ent.text, ent.label_)
```

<a id="nltk"></a>
### NLTK

- Docs: https://www.nltk.org/
- GitHub: https://github.com/nltk/nltk
- PyPI: https://pypi.org/project/nltk/

```python
from nltk import pos_tag, word_tokenize

# Tokenize and POS-tag text
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
text = "Python powers many ML workflows."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print(tokens)
print(tagged)
```

<a id="openai-python-sdk"></a>
### OpenAI Python SDK

- Docs: https://platform.openai.com/docs/libraries/python
- GitHub: https://github.com/openai/openai-python
- PyPI: https://pypi.org/project/openai/

```python
from openai import OpenAI

# Chat completion call (requires OPENAI_API_KEY)
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain gradient descent in one sentence."},
    ],
)

print(response.choices[0].message.content)
```

<a id="langchain"></a>
### LangChain

- Docs: https://python.langchain.com/docs/introduction/
- GitHub: https://github.com/langchain-ai/langchain
- PyPI: https://pypi.org/project/langchain/

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Simple prompt + model chain
prompt = ChatPromptTemplate.from_template("Summarize this topic in one line: {topic}")
llm = ChatOpenAI(model="gpt-4.1-mini")
chain = prompt | llm

print(chain.invoke({"topic": "overfitting"}).content)
```

<a id="llamaindex"></a>
### LlamaIndex

- Docs: https://docs.llamaindex.ai/
- GitHub: https://github.com/run-llama/llama_index
- PyPI: https://pypi.org/project/llama-index/

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Basic RAG setup from local docs
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

print(query_engine.query("What is this project about?"))
```

---

<a id="sec6"></a>
## 6. 👁️ Computer Vision

<a id="opencv"></a>
### OpenCV

- Docs: https://docs.opencv.org/
- GitHub: https://github.com/opencv/opencv
- PyPI: https://pypi.org/project/opencv-python/

```python
import cv2

# Read image, resize, and detect edges
img = cv2.imread("input.jpg")
resized = cv2.resize(img, (320, 240))
edges = cv2.Canny(resized, 100, 200)

cv2.imwrite("edges.jpg", edges)
```

<a id="torchvision"></a>
### TorchVision

- Docs: https://pytorch.org/vision/stable/index.html
- GitHub: https://github.com/pytorch/vision
- PyPI: https://pypi.org/project/torchvision/

```python
import torch
from PIL import Image
from torchvision import models, transforms

# Pretrained model inference
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img = Image.open("image.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)

pred_idx = int(logits.argmax(dim=1))
print(pred_idx)
```

<a id="detectron2"></a>
### Detectron2

Detectron2 is Meta's high-performance framework for object detection and segmentation research.

- Docs: https://detectron2.readthedocs.io/
- GitHub: https://github.com/facebookresearch/detectron2
- PyPI: https://pypi.org/project/detectron2/

<a id="mmdetection-mmengine"></a>
### MMDetection / MMEngine

MMDetection and MMEngine provide modular training/inference stacks for detection and broader model lifecycle orchestration.

- MMDetection Docs: https://mmdetection.readthedocs.io/
- MMDetection GitHub: https://github.com/open-mmlab/mmdetection
- MMDetection PyPI: https://pypi.org/project/mmdet/
- MMEngine Docs: https://mmengine.readthedocs.io/
- MMEngine GitHub: https://github.com/open-mmlab/mmengine
- MMEngine PyPI: https://pypi.org/project/mmengine/

<a id="mediapipe"></a>
### mediapipe

- Docs: https://ai.google.dev/edge/mediapipe/solutions/guide
- GitHub: https://github.com/google-ai-edge/mediapipe
- PyPI: https://pypi.org/project/mediapipe/

```python
import cv2
import mediapipe as mp

# Hand tracking on a single frame
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

image = cv2.imread("hand.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = hands.process(rgb)

if result.multi_hand_landmarks:
    print(f"Detected hands: {len(result.multi_hand_landmarks)}")
else:
    print("No hands detected")

hands.close()
```

---

<a id="sec7"></a>
## 7. 📊 Visualization

<a id="matplotlib"></a>
### Matplotlib

- Docs: https://matplotlib.org/stable/
- GitHub: https://github.com/matplotlib/matplotlib
- PyPI: https://pypi.org/project/matplotlib/

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot + subplot layout
x = np.linspace(0, 2 * np.pi, 200)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, y1, label="sin(x)")
axes[0].legend()
axes[1].plot(x, y2, color="orange", label="cos(x)")
axes[1].legend()
plt.tight_layout()
plt.show()
```

<a id="seaborn"></a>
### Seaborn

- Docs: https://seaborn.pydata.org/
- GitHub: https://github.com/mwaskom/seaborn
- PyPI: https://pypi.org/project/seaborn/

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap + pairplot for quick EDA
tips = sns.load_dataset("tips")
num = tips[["total_bill", "tip", "size"]]
sns.heatmap(num.corr(), annot=True, cmap="viridis")
plt.show()

sns.pairplot(tips[["total_bill", "tip", "size"]])
plt.show()
```

<a id="plotly"></a>
### Plotly

- Docs: https://plotly.com/python/
- GitHub: https://github.com/plotly/plotly.py
- PyPI: https://pypi.org/project/plotly/

```python
import pandas as pd
import plotly.express as px

# Interactive scatter plot
df = pd.DataFrame(
    {
        "x": [1, 2, 3, 4],
        "y": [2, 1, 3, 4],
        "label": ["A", "B", "C", "D"],
    }
)
fig = px.scatter(df, x="x", y="y", text="label", title="Interactive Scatter")
fig.show()
```

<a id="tensorboard"></a>
### TensorBoard

- Docs: https://www.tensorflow.org/tensorboard
- GitHub: https://github.com/tensorflow/tensorboard
- PyPI: https://pypi.org/project/tensorboard/

```python
from torch.utils.tensorboard import SummaryWriter

# Log metrics for visualization in TensorBoard
writer = SummaryWriter(log_dir="runs/exp1")
for step in range(10):
    writer.add_scalar("train/loss", 1.0 / (step + 1), step)
writer.close()
```

<a id="weights--biases-wandb"></a>
### Weights & Biases (wandb)

- Docs: https://docs.wandb.ai/
- GitHub: https://github.com/wandb/wandb
- PyPI: https://pypi.org/project/wandb/

```python
import wandb

# Track an experiment and log metrics
wandb.init(project="ml-handbook-demo", config={"lr": 1e-3, "epochs": 5})
for epoch in range(5):
    wandb.log({"epoch": epoch, "accuracy": 0.7 + epoch * 0.05})
wandb.finish()
```

---

<a id="sec8"></a>
## 8. 🧬 Specialized AI Domains

<a id="time-series"></a>
### Time Series

#### Prophet
- Docs: https://facebook.github.io/prophet/
- GitHub: https://github.com/facebook/prophet
- PyPI: https://pypi.org/project/prophet/

```python
import pandas as pd
from prophet import Prophet

# Forecast a univariate time series
df = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=30), "y": range(30)})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
print(forecast[["ds", "yhat"]].tail())
```

#### GluonTS
- Docs: https://ts.gluon.ai/
- GitHub: https://github.com/awslabs/gluonts
- PyPI: https://pypi.org/project/gluonts/

GluonTS provides probabilistic forecasting models and evaluation utilities built around time-series datasets.

#### Kats
- Docs: https://facebookresearch.github.io/Kats/
- GitHub: https://github.com/facebookresearch/Kats
- PyPI: https://pypi.org/project/kats/

Kats offers production-focused components for anomaly detection, forecasting, and decomposition.

<a id="reinforcement-learning"></a>
### Reinforcement Learning

#### Stable-Baselines3
- Docs: https://stable-baselines3.readthedocs.io/
- GitHub: https://github.com/DLR-RM/stable-baselines3
- PyPI: https://pypi.org/project/stable-baselines3/

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Train a PPO agent on CartPole
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1_000)
obs, _ = env.reset()
action, _ = model.predict(obs)
print(action)
```

#### RLlib
- Docs: https://docs.ray.io/en/latest/rllib/
- GitHub: https://github.com/ray-project/ray
- PyPI: https://pypi.org/project/ray/

RLlib (part of Ray) supports scalable distributed reinforcement learning across many algorithms.

#### Gymnasium
- Docs: https://gymnasium.farama.org/
- GitHub: https://github.com/Farama-Foundation/Gymnasium
- PyPI: https://pypi.org/project/gymnasium/

Gymnasium provides standardized RL environments and APIs used by most RL frameworks.

<a id="audio--speech"></a>
### Audio / Speech

#### torchaudio
- Docs: https://pytorch.org/audio/stable/index.html
- GitHub: https://github.com/pytorch/audio
- PyPI: https://pypi.org/project/torchaudio/

```python
import torchaudio

# Load waveform and inspect metadata
waveform, sample_rate = torchaudio.load("sample.wav")
print(waveform.shape, sample_rate)
```

#### librosa
- Docs: https://librosa.org/doc/latest/index.html
- GitHub: https://github.com/librosa/librosa
- PyPI: https://pypi.org/project/librosa/

```python
import librosa

# Load audio and compute MFCCs
y, sr = librosa.load("sample.wav", sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(mfcc.shape)
```

#### SpeechBrain
- Docs: https://speechbrain.readthedocs.io/
- GitHub: https://github.com/speechbrain/speechbrain
- PyPI: https://pypi.org/project/speechbrain/

SpeechBrain offers ready-to-use speech pipelines for ASR, speaker recognition, and enhancement.

<a id="graph-ml"></a>
### Graph ML

#### PyTorch Geometric (PyG)
- Docs: https://pytorch-geometric.readthedocs.io/
- GitHub: https://github.com/pyg-team/pytorch_geometric
- PyPI: https://pypi.org/project/torch-geometric/

```python
import torch
from torch_geometric.data import Data

# Define a tiny graph with node features
x = torch.tensor([[1.0], [2.0], [3.0]])
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
graph = Data(x=x, edge_index=edge_index)
print(graph)
```

#### Deep Graph Library (DGL)
- Docs: https://docs.dgl.ai/
- GitHub: https://github.com/dmlc/dgl
- PyPI: https://pypi.org/project/dgl/

DGL provides high-performance graph data structures and message-passing APIs for graph neural networks.

---

<a id="sec9"></a>
## 9. 🧩 Core Essentials Stack

<a id="minimum-viable-stack"></a>
### Minimum Viable Stack

The six-library starter stack that covers most practical beginner-to-intermediate ML workflows:

1. NumPy
2. Pandas
3. Matplotlib
4. scikit-learn
5. PyTorch
6. HuggingFace Transformers

| Library | Core Role |
|---|---|
| NumPy | Numerical arrays, linear algebra |
| Pandas | Tabular data manipulation |
| Matplotlib | Data visualization |
| scikit-learn | Classical ML models, preprocessing, metrics |
| PyTorch | Deep learning model development |
| HuggingFace Transformers | Pretrained transformer models for text/vision/multimodal tasks |

<a id="day-1-script"></a>
### Day 1 Script

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import pipeline

# 1) NumPy + Pandas: create a tiny synthetic dataset
X_np = np.random.randn(200, 2)
y_np = (X_np[:, 0] + X_np[:, 1] > 0).astype(int)
df = pd.DataFrame(X_np, columns=["f1", "f2"])
df["target"] = y_np

# 2) Matplotlib: quick visualization
plt.scatter(df["f1"], df["f2"], c=df["target"], cmap="coolwarm", alpha=0.7)
plt.title("Day 1 Dataset")
plt.show()

# 3) scikit-learn: train a baseline classifier
X_train, X_test, y_train, y_test = train_test_split(df[["f1", "f2"]], df["target"], test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("sklearn accuracy:", clf.score(X_test, y_test))

# 4) PyTorch: tensor operation
tensor = torch.tensor(X_np[:5], dtype=torch.float32)
print("torch tensor mean:", tensor.mean(dim=0))

# 5) Transformers: run a sentiment pipeline
sentiment = pipeline("sentiment-analysis")
print(sentiment("I am learning ML step by step."))
```

---

<a id="sec10"></a>
## 10. 📚 Full Resource Reference

| Library | Official Docs | GitHub | PyPI | One-line Description |
|---|---|---|---|---|
| NumPy | https://numpy.org/doc/ | https://github.com/numpy/numpy | https://pypi.org/project/numpy/ | Core numerical arrays and linear algebra foundation for Python ML. |
| SciPy | https://docs.scipy.org/doc/scipy/ | https://github.com/scipy/scipy | https://pypi.org/project/scipy/ | Scientific algorithms for optimization, signal processing, and stats. |
| Numba | https://numba.readthedocs.io/ | https://github.com/numba/numba | https://pypi.org/project/numba/ | JIT compiler that accelerates Python numeric code. |
| CuPy | https://docs.cupy.dev/ | https://github.com/cupy/cupy | https://pypi.org/project/cupy-cuda12x/ | GPU array library with a NumPy-compatible API. |
| scikit-learn | https://scikit-learn.org/stable/ | https://github.com/scikit-learn/scikit-learn | https://pypi.org/project/scikit-learn/ | Standard classical ML toolkit for tabular data and pipelines. |
| XGBoost | https://xgboost.readthedocs.io/ | https://github.com/dmlc/xgboost | https://pypi.org/project/xgboost/ | Optimized gradient boosting for high-performance tabular ML. |
| LightGBM | https://lightgbm.readthedocs.io/ | https://github.com/microsoft/LightGBM | https://pypi.org/project/lightgbm/ | Fast gradient boosting framework for large datasets. |
| CatBoost | https://catboost.ai/en/docs/ | https://github.com/catboost/catboost | https://pypi.org/project/catboost/ | Boosting framework with strong categorical feature handling. |
| StatsModels | https://www.statsmodels.org/stable/index.html | https://github.com/statsmodels/statsmodels | https://pypi.org/project/statsmodels/ | Statistical modeling and inference library for Python. |
| PyTorch | https://pytorch.org/docs/stable/index.html | https://github.com/pytorch/pytorch | https://pypi.org/project/torch/ | Flexible deep learning framework popular in research and production. |
| TensorFlow | https://www.tensorflow.org/api_docs | https://github.com/tensorflow/tensorflow | https://pypi.org/project/tensorflow/ | End-to-end deep learning ecosystem from Google. |
| Keras | https://keras.io/ | https://github.com/keras-team/keras | https://pypi.org/project/keras/ | High-level neural network API, tightly integrated with TensorFlow. |
| JAX | https://jax.readthedocs.io/ | https://github.com/jax-ml/jax | https://pypi.org/project/jax/ | High-performance autodiff and compilation framework. |
| MindSpore | https://www.mindspore.cn/en | https://github.com/mindspore-ai/mindspore | https://pypi.org/project/mindspore/ | Huawei framework for AI training and deployment. |
| MXNet | https://mxnet.apache.org/ | https://github.com/apache/mxnet | https://pypi.org/project/mxnet/ | Apache deep learning framework used in some legacy stacks. |
| Pandas | https://pandas.pydata.org/docs/ | https://github.com/pandas-dev/pandas | https://pypi.org/project/pandas/ | DataFrame-based data wrangling and analytics toolkit. |
| Polars | https://docs.pola.rs/ | https://github.com/pola-rs/polars | https://pypi.org/project/polars/ | Fast Rust-powered dataframe engine for Python. |
| Dask | https://docs.dask.org/ | https://github.com/dask/dask | https://pypi.org/project/dask/ | Parallel and distributed computing for Python data workflows. |
| PyArrow | https://arrow.apache.org/docs/python/ | https://github.com/apache/arrow | https://pypi.org/project/pyarrow/ | Arrow/Parquet columnar data format and memory interoperability. |
| HuggingFace Datasets | https://huggingface.co/docs/datasets | https://github.com/huggingface/datasets | https://pypi.org/project/datasets/ | Dataset loading and processing library for ML/LLM tasks. |
| HuggingFace Transformers | https://huggingface.co/docs/transformers | https://github.com/huggingface/transformers | https://pypi.org/project/transformers/ | Pretrained transformer models for NLP, vision, and multimodal AI. |
| SentenceTransformers | https://www.sbert.net/ | https://github.com/UKPLab/sentence-transformers | https://pypi.org/project/sentence-transformers/ | Sentence embeddings for semantic search and similarity. |
| spaCy | https://spacy.io/usage | https://github.com/explosion/spaCy | https://pypi.org/project/spacy/ | Industrial-strength NLP pipelines and components. |
| NLTK | https://www.nltk.org/ | https://github.com/nltk/nltk | https://pypi.org/project/nltk/ | Classic NLP toolkit for tokenization, tagging, and corpora. |
| OpenAI Python SDK | https://platform.openai.com/docs/libraries/python | https://github.com/openai/openai-python | https://pypi.org/project/openai/ | Python client for OpenAI model APIs. |
| LangChain | https://python.langchain.com/docs/introduction/ | https://github.com/langchain-ai/langchain | https://pypi.org/project/langchain/ | Framework for composing LLM apps, chains, and tools. |
| LlamaIndex | https://docs.llamaindex.ai/ | https://github.com/run-llama/llama_index | https://pypi.org/project/llama-index/ | Data framework focused on indexing and RAG pipelines. |
| OpenCV | https://docs.opencv.org/ | https://github.com/opencv/opencv | https://pypi.org/project/opencv-python/ | Core computer vision and image processing library. |
| TorchVision | https://pytorch.org/vision/stable/index.html | https://github.com/pytorch/vision | https://pypi.org/project/torchvision/ | Vision datasets, transforms, and pretrained models for PyTorch. |
| Detectron2 | https://detectron2.readthedocs.io/ | https://github.com/facebookresearch/detectron2 | https://pypi.org/project/detectron2/ | Advanced object detection and segmentation framework. |
| MMDetection | https://mmdetection.readthedocs.io/ | https://github.com/open-mmlab/mmdetection | https://pypi.org/project/mmdet/ | Modular object detection toolbox from OpenMMLab. |
| MMEngine | https://mmengine.readthedocs.io/ | https://github.com/open-mmlab/mmengine | https://pypi.org/project/mmengine/ | Training/inference engine used across OpenMMLab projects. |
| mediapipe | https://ai.google.dev/edge/mediapipe/solutions/guide | https://github.com/google-ai-edge/mediapipe | https://pypi.org/project/mediapipe/ | Real-time cross-platform perception pipelines (hands/pose/face). |
| Matplotlib | https://matplotlib.org/stable/ | https://github.com/matplotlib/matplotlib | https://pypi.org/project/matplotlib/ | Foundational plotting library for static charts. |
| Seaborn | https://seaborn.pydata.org/ | https://github.com/mwaskom/seaborn | https://pypi.org/project/seaborn/ | Statistical plotting API built on Matplotlib. |
| Plotly | https://plotly.com/python/ | https://github.com/plotly/plotly.py | https://pypi.org/project/plotly/ | Interactive plotting library for browser-based visuals. |
| TensorBoard | https://www.tensorflow.org/tensorboard | https://github.com/tensorflow/tensorboard | https://pypi.org/project/tensorboard/ | Training visualization dashboard for metrics and graphs. |
| wandb | https://docs.wandb.ai/ | https://github.com/wandb/wandb | https://pypi.org/project/wandb/ | Experiment tracking and model observability platform. |
| Prophet | https://facebook.github.io/prophet/ | https://github.com/facebook/prophet | https://pypi.org/project/prophet/ | Time-series forecasting with trend/seasonality decomposition. |
| GluonTS | https://ts.gluon.ai/ | https://github.com/awslabs/gluonts | https://pypi.org/project/gluonts/ | Probabilistic deep learning toolkit for forecasting. |
| Kats | https://facebookresearch.github.io/Kats/ | https://github.com/facebookresearch/Kats | https://pypi.org/project/kats/ | Time-series analysis framework for forecasting and anomalies. |
| Stable-Baselines3 | https://stable-baselines3.readthedocs.io/ | https://github.com/DLR-RM/stable-baselines3 | https://pypi.org/project/stable-baselines3/ | Reliable implementations of common RL algorithms. |
| RLlib | https://docs.ray.io/en/latest/rllib/ | https://github.com/ray-project/ray | https://pypi.org/project/ray/ | Scalable distributed reinforcement learning library in Ray. |
| Gymnasium | https://gymnasium.farama.org/ | https://github.com/Farama-Foundation/Gymnasium | https://pypi.org/project/gymnasium/ | Standard API and benchmark environments for RL research. |
| torchaudio | https://pytorch.org/audio/stable/index.html | https://github.com/pytorch/audio | https://pypi.org/project/torchaudio/ | Audio I/O and transforms for speech and sound ML. |
| librosa | https://librosa.org/doc/latest/index.html | https://github.com/librosa/librosa | https://pypi.org/project/librosa/ | Audio/music analysis library for feature extraction. |
| SpeechBrain | https://speechbrain.readthedocs.io/ | https://github.com/speechbrain/speechbrain | https://pypi.org/project/speechbrain/ | Open speech toolkit with pretrained models and recipes. |
| PyTorch Geometric (PyG) | https://pytorch-geometric.readthedocs.io/ | https://github.com/pyg-team/pytorch_geometric | https://pypi.org/project/torch-geometric/ | Graph neural network library for PyTorch. |
| Deep Graph Library (DGL) | https://docs.dgl.ai/ | https://github.com/dmlc/dgl | https://pypi.org/project/dgl/ | High-performance graph ML framework across backends. |

---

### Notes

- Some examples require datasets, model weights, or API keys.
- GPU examples (CuPy, PyTorch CUDA workflows) require compatible hardware/software.
- Pin versions in a real project for reproducibility.
