# Python AI/ML Engineering Tooling

A reference inventory of commonly used Python tools, libraries, and platforms for AI/ML engineering. Organized by workflow stage so it can later map to Rust equivalents or gaps.

---

## Core Numerical & Scientific Computing

| Tool | Role |
|------|------|
| **NumPy** | N-dimensional arrays, linear algebra, broadcasting |
| **SciPy** | Scientific algorithms (optimization, stats, signal processing) |
| **Pandas** | Tabular dataframes, ETL, time series |
| **Polars** | Fast dataframe library (Rust-backed, Python API) |
| **PyArrow** | Columnar in-memory format (Arrow), Parquet I/O |
| **Dask** | Parallel/distributed NumPy/Pandas-style compute |
| **CuPy** | NumPy-compatible arrays on NVIDIA GPUs |
| **Numba** | JIT compilation for numerical Python |
| **Cython** | Compile Python/C extensions for speed |
| **JAX** | Autograd + XLA; NumPy-like functional ML |

---

## Deep Learning Frameworks

| Tool | Role |
|------|------|
| **PyTorch** | Dominant research/production DL framework |
| **TensorFlow / Keras** | End-to-end DL; Keras high-level API |
| **JAX / Flax / Haiku** | Functional, high-performance training stacks |
| **Lightning (PyTorch Lightning)** | Structured training loops, multi-GPU/TPU |
| **Ignite** | Lightweight PyTorch training helpers |
| **Accelerate (Hugging Face)** | Multi-device training abstraction |
| **DeepSpeed** | Large-model training (ZeRO, offload, etc.) |
| **FSDP / Megatron-LM** | Sharded / large-scale LLM training |
| **ONNX / ONNX Runtime** | Cross-framework model interchange & inference |
| **TorchScript / torch.compile** | Graph capture and optimized PyTorch execution |
| **TensorRT** | NVIDIA inference optimization |
| **OpenVINO** | Intel inference optimization |
| **MLX** | Apple Silicon ML framework (Python API) |

---

## Classical Machine Learning

| Tool | Role |
|------|------|
| **scikit-learn** | Classification, regression, clustering, pipelines |
| **XGBoost** | Gradient boosted trees |
| **LightGBM** | Fast gradient boosting |
| **CatBoost** | Gradient boosting with categorical support |
| **statsmodels** | Statistical models, hypothesis testing |
| **imbalanced-learn** | Resampling for class imbalance |
| **Optuna** | Hyperparameter optimization |
| **Hyperopt** | Bayesian/TPE hyperparameter search |
| **Ray Tune** | Distributed hyperparameter tuning |

---

## NLP & Transformers / LLMs

| Tool | Role |
|------|------|
| **Hugging Face Transformers** | Pretrained models, tokenizers, pipelines |
| **tokenizers** | Fast tokenization (Rust-backed) |
| **datasets** | Dataset loading, streaming, processing |
| **Evaluate** | NLP/ML metrics |
| **PEFT** | Parameter-efficient fine-tuning (LoRA, etc.) |
| **TRL** | RLHF / preference tuning helpers |
| **sentence-transformers** | Embedding models and similarity |
| **spaCy** | Production NLP pipelines |
| **NLTK** | Classical NLP utilities |
| **Gensim** | Topic modeling, word embeddings |
| **OpenAI / Anthropic / Google SDKs** | Hosted LLM APIs |
| **LiteLLM** | Unified multi-provider LLM API |
| **vLLM** | High-throughput LLM serving |
| **llama.cpp Python bindings** | Local GGUF inference |
| **Ollama (Python client)** | Local model runner integration |
| **Guidance / Outlines / Instructor** | Structured / constrained LLM output |
| **LangChain** | LLM app orchestration, tools, chains |
| **LlamaIndex** | RAG frameworks and data connectors |
| **Haystack** | NLP/RAG pipelines |
| **Semantic Kernel** | Agent/orchestration SDK |
| **DSPy** | Programmatic prompt/optimizer workflows |
| **AutoGen / CrewAI** | Multi-agent frameworks |

---

## Computer Vision

| Tool | Role |
|------|------|
| **OpenCV (cv2)** | Image/video processing |
| **Pillow (PIL)** | Image I/O and transforms |
| **torchvision** | Vision models and datasets |
| **timm** | Image model zoo |
| **Albumentations** | Fast image augmentation |
| **Detectron2** | Detection/segmentation |
| **Ultralytics (YOLO)** | Object detection training/inference |
| **MMDetection / MMSegmentation** | OpenMMLab detection/seg toolkits |
| **kornia** | Differentiable CV operators |
| **scikit-image** | Classical image processing |

---

## Audio / Speech / Multimodal

| Tool | Role |
|------|------|
| **torchaudio** | Audio I/O and transforms |
| **librosa** | Audio analysis |
| **whisper / faster-whisper** | Speech-to-text |
| **SpeechBrain** | Speech toolkit |
| **Coqui TTS / Tortoise** | Text-to-speech |
| **PyAV** | FFmpeg bindings for media |
| **OpenAI Whisper API clients** | Hosted ASR |

---

## Data Labeling, Versioning & Feature Stores

| Tool | Role |
|------|------|
| **Label Studio** | Annotation UI for text/image/audio |
| **CVAT** | Computer vision annotation |
| **Prodigy** | Active learning annotation (spaCy ecosystem) |
| **DVC** | Data/model versioning with Git-like workflows |
| **LakeFS** | Data lake versioning |
| **Pachyderm** | Data lineage pipelines |
| **Feast** | Feature store |
| **Tecton** | Managed feature platform |
| **Delta Lake / Iceberg (Python APIs)** | Table formats for ML data |

---

## Experiment Tracking & Model Registry

| Tool | Role |
|------|------|
| **MLflow** | Tracking, projects, model registry |
| **Weights & Biases (wandb)** | Experiments, artifacts, sweeps |
| **TensorBoard** | Metrics and graph visualization |
| **Neptune** | Experiment tracking |
| **Comet** | Experiment management |
| **Aim** | Open-source experiment tracking |
| **ClearML** | Tracking + orchestration |
| **Sacred / Hydra** | Config and experiment management |
| **DVCLive** | Metrics logging with DVC |

---

## Workflow Orchestration & Pipelines

| Tool | Role |
|------|------|
| **Airflow** | DAG-based batch orchestration |
| **Prefect** | Modern Python-native workflows |
| **Dagster** | Data-aware pipeline orchestration |
| **Kubeflow Pipelines** | K8s ML pipelines |
| **Metaflow** | Human-centric ML workflows (Netflix) |
| **ZenML** | MLOps pipeline abstraction |
| **Flyte** | Typed, scalable workflows |
| **Luigi** | Batch pipeline dependency graphs |
| **Kedro** | Project structure + pipelines |
| **Ray** | Distributed compute for training/serving |

---

## Model Serving & APIs

| Tool | Role |
|------|------|
| **FastAPI** | High-performance inference APIs |
| **Flask / Django** | General web APIs wrapping models |
| **BentoML** | Package and serve ML services |
| **TorchServe** | PyTorch model serving |
| **TensorFlow Serving** | TF model serving |
| **Triton Inference Server (Python clients)** | Multi-framework GPU serving |
| **Seldon Core** | K8s model deployment |
| **KServe** | K8s serverless inference |
| **Gradio** | Quick ML demos/UIs |
| **Streamlit** | Data/ML app UIs |
| **NiceGUI / Panel / Dash** | Interactive analytics UIs |
| **Modal / RunPod / Replicate SDKs** | Serverless/GPU cloud inference |

---

## Vector Databases & RAG Infrastructure

| Tool | Role |
|------|------|
| **FAISS** | Similarity search (Meta) |
| **Annoy / HNSWlib** | Approximate nearest neighbors |
| **Chroma** | Embedded vector DB |
| **Qdrant (Python client)** | Vector search engine |
| **Weaviate (client)** | Vector + hybrid search |
| **Pinecone (client)** | Managed vector DB |
| **Milvus / pymilvus** | Scalable vector DB |
| **pgvector** | Postgres vector extension (Python drivers) |
| **LanceDB** | Embedded multimodal vector DB |
| **Redis / Elasticsearch vector features** | Hybrid search backends |

---

## Evaluation, Testing & Observability

| Tool | Role |
|------|------|
| **pytest** | Unit/integration testing |
| **Great Expectations / Pandera** | Data validation |
| **Deepchecks** | ML data/model validation |
| **Evidently** | Drift and model quality monitoring |
| **WhyLabs / whylogs** | Data logging and monitoring |
| **Ragas** | RAG evaluation |
| **DeepEval** | LLM evaluation framework |
| **LangSmith** | LLM tracing and eval |
| **Phoenix (Arize)** | LLM observability |
| **OpenTelemetry Python** | Tracing/metrics for services |
| **Promptfoo (Python usage)** | Prompt regression testing |
| **Giskard** | ML/LLM testing and scanning |

---

## Interpretability & Fairness

| Tool | Role |
|------|------|
| **SHAP** | Feature attribution |
| **LIME** | Local model explanations |
| **Captum** | PyTorch interpretability |
| **eli5** | Debug/explain ML models |
| **Fairlearn** | Fairness metrics and mitigation |
| **AIF360** | Bias detection/mitigation |
| **What-If Tool** | Interactive model probing |
| **Alibi / Alibi Detect** | Explainability and outlier/drift detection |

---

## Notebooks, IDE & Developer Experience

| Tool | Role |
|------|------|
| **Jupyter / JupyterLab** | Interactive notebooks |
| **IPython** | Enhanced REPL |
| **VS Code / Cursor Python extensions** | Editing, debugging, notebooks |
| **PyCharm** | Full IDE |
| **Marimo** | Reactive Python notebooks |
| **Papermill** | Parameterized notebook execution |
| **nbconvert / jupytext** | Notebook conversion/sync |
| **Rich / tqdm** | Terminal UX and progress bars |
| **ipywidgets** | Interactive notebook widgets |

---

## Packaging, Environments & Dependency Management

| Tool | Role |
|------|------|
| **pip / PyPI** | Package install and registry |
| **conda / mamba / micromamba** | Env + binary dependency management |
| **uv** | Fast Python package/env manager |
| **poetry** | Dependency and packaging |
| **pdm** | PEP 621 packaging |
| **virtualenv / venv** | Isolated environments |
| **pip-tools** | Lock files from requirements |
| **Docker / docker-compose** | Reproducible runtime images |
| **Hatch / setuptools / flit** | Build backends |

---

## Cloud, GPU & Infrastructure SDKs

| Tool | Role |
|------|------|
| **boto3** | AWS SDK |
| **google-cloud-\*** | GCP SDKs (Vertex, Storage, etc.) |
| **azure-ai-\*** | Azure AI/ML SDKs |
| **sagemaker SDK** | AWS training/deploy |
| **Vertex AI SDK** | GCP managed ML |
| **Kubernetes Python client** | Cluster automation |
| **NVIDIA CUDA Python / cuDNN bindings** | GPU stack access |
| **pynvml** | GPU monitoring |
| **Slurm / submitit** | HPC job submission |

---

## Synthetic Data, Simulation & RL

| Tool | Role |
|------|------|
| **Gymnasium (Gym)** | RL environment API |
| **Stable-Baselines3** | RL algorithms |
| **RLlib** | Distributed RL (Ray) |
| **PettingZoo** | Multi-agent RL envs |
| **MuJoCo / PyBullet** | Physics simulation |
| **Unity ML-Agents** | Game/sim RL |
| **Faker / SDV / Gretel** | Synthetic tabular/data generation |
| **snorkel** | Programmatic weak labeling |

---

## Serialization, Config & Utilities

| Tool | Role |
|------|------|
| **pickle / joblib** | Model/object serialization |
| **safetensors** | Safe tensor weight format |
| **HDF5 / h5py** | Large array storage |
| **zarr** | Chunked N-dimensional storage |
| **OmegaConf / Hydra** | Hierarchical configs |
| **Pydantic** | Data validation and settings |
| **Typer / Click / argparse** | CLI tooling |
| **loguru** | Ergonomic logging |
| **einops** | Tensor rearrange/ops sugar |
| **more-itertools / toolz** | Functional utilities |

---

## Security, Safety & Compliance (ML-specific)

| Tool | Role |
|------|------|
| **Presidio** | PII detection/anonymization |
| **Art (Adversarial Robustness Toolbox)** | Adversarial attacks/defenses |
| **Privacy Ratchet / Opacus** | Differential privacy training |
| **LLM Guard / NeMo Guardrails** | LLM input/output safety |
| **Bandit** | Python security linting (general) |

---

## Notes for Rust AI Tooling Mapping

When designing Rust counterparts, the highest-leverage Python categories for AI *development* (not just inference) tend to be:

1. **Training & autodiff frameworks** (PyTorch/JAX equivalents)
2. **Tokenizers, datasets, and model hubs** (Transformers ecosystem)
3. **Experiment tracking & reproducibility** (MLflow/W&B/DVC)
4. **Serving & local LLM runtimes** (vLLM, llama.cpp-class tools)
5. **RAG/agent orchestration** (LangChain/LlamaIndex-class libraries)
6. **Evaluation & observability for LLMs** (Ragas, LangSmith, Phoenix)
7. **Packaging/DX** (uv/poetry-class ergonomics for ML projects)

This list is not exhaustive of every niche package, but covers tooling that is **commonly used in professional AI/ML engineering workflows** as of 2026.
