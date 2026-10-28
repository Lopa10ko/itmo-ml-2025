# ITMO ML · 2025-2027

Coursework, experiments, and lab reports from machine learning studies at ITMO University. The repository follows the path from data preparation and classical models to deep learning, time series, and automated model selection.

Most lab reports are Jupyter notebooks combining code, explanations, visualizations, and experimental results. Assignment PDFs, datasets, scripts, and exported reports sit alongside them. Materials are in English and Russian; lab numbering follows each course independently.

## Courses at a glance

| Course | Focus | Materials |
| --- | --- | --- |
| [Machine Learning](#machine-learning) | Applied ML: tabular prediction, clustering, forecasting, computer vision, and transformers | [`ml/`](ml/) · labs 1–12, with supporting artifacts only for labs 7–8 |
| [Mathematical Foundations of Machine Learning](#mathematical-foundations-of-machine-learning) | Linear algebra, numerical optimization, multivariate analysis, and probabilistic modeling | [`mfml/`](mfml/) · 4 labs and team reports |
| [Deep Learning](#deep-learning) | CNNs, language model fine-tuning, agent systems, and generative models | [`dl/`](dl/) · 4 labs |
| [Machine Learning for Time Series](#machine-learning-for-time-series) | Signal characterization, forecasting, causal analysis, reconstructed dynamics, and event processes | [`tsml/`](tsml/) · 8 labs |
| [Automated Machine Learning](#automated-machine-learning) | Hyperparameter optimization, meta-features, and algorithm selection | [`automl/`](automl/) · 2 labs |

## Machine Learning

An applied sequence covering the full modeling workflow: exploratory analysis, feature engineering, validation, model comparison, and interpretation. Later labs move from tabular data to neural networks, object detection, attention, and image generation.

| Lab | What it is about | Artifacts |
| --- | --- | --- |
| 1 · Apartment prices | Clean housing data, engineer features, study price drivers, and build a price prediction model. | [Notebook](ml/lab-1/334771_lab_1_apartment_analysis.ipynb) · [Data](ml/lab-1/data_lab_1.csv) |
| 2 · Bank customer churn | Predict customer departure and evaluate classification under class imbalance using F1 and ROC AUC. | [Notebook](ml/lab-2/334771_lab_2_bank_analysis.ipynb) · [Data](ml/lab-2/Bank_data.csv) |
| 3 · Countries and charity | Compare k-means, hierarchical clustering, and DBSCAN to identify groups of countries needing assistance. | [Notebook](ml/lab-3/334771_lab_3_countries_charity.ipynb) · [Data](ml/lab-3/Lab3.csv) |
| 4 · Credit scoring | Compare decision trees and ensembles, tune their parameters, and inspect feature importance. | [Notebook](ml/lab-4/334771_lab_4_credit_scoring%20%282%29.ipynb) · [Decision tree](ml/lab-4/decision_tree.svg) · [Charts](ml/lab-4/) |
| 5 · Hospitality employment | Analyze trend, seasonality, and stationarity, then forecast monthly California hospitality employment. | [Notebook](ml/lab-5/334771_lab_5_california_hospitality_forecasting%20%281%29.ipynb) · [Data](ml/lab-5/california_hospitality.csv) |
| 6 · Movie ratings | Join film and box-office records, prepare features, and train a neural network to predict movie ratings. | [Notebook](ml/lab-6/334771_lab_6_mkrf_movie_rating.ipynb) · [Data files](ml/lab-6/) |
| 7 · Supporting data | Credit-scoring dataset; no lab notebook is included in this folder. | [Data](ml/lab-7/credit_scoring.csv) |
| 8 · EuroSAT artifacts | Remote-sensing dataset reference and a saved ResNet checkpoint; no lab notebook is included in this folder. | [Dataset submodule](ml/lab-8/EuroSAT) · [Checkpoint](ml/lab-8/best_kfold_resnet_eurocat.pth) |
| 9 · Brain tumor detection | Apply YOLO to brain tumor detection in medical images. | [Notebook](ml/lab-9/334771_lab_9_yolo_brain_tumor.ipynb) |
| 10 · Text transformer | Implement multi-head attention, positional encoding, and a transformer encoder for text classification. | [Notebook](ml/lab-10/334771_lab_10_attention.ipynb) · [Saved model](ml/lab-10/text_model.pth) |
| 11 · Vision and language | Build a vision transformer and explore contrastive image–text learning and zero-shot classification. | [Notebook](ml/lab-11/334771_lab_11_attention.ipynb) |
| 12 · Painting restoration | A ComfyUI workflow for image restoration and generation with reference conditioning and upscaling. | [Workflow JSON](ml/lab-12/334771_lab_12_painting_restoration.json) · [Input image](ml/lab-12/adm_dressing_1.-.jpg) · [Output image](ml/lab-12/ComfyUI_00001.png) |

## Mathematical Foundations of Machine Learning

Team 7's labs connect mathematical methods to practical modeling: matrix decompositions for representation learning, optimization algorithms for fitting models, and multivariate and Bayesian analysis of medical appointment attendance.

| Lab | What it is about | Artifacts |
| --- | --- | --- |
| 1 · Matrix decompositions | Implement SVD, PCA, and kernel PCA; compare representations and their effect on downstream classification. | [Notebook](mfml/lab-1/MFML_Team_7_LAB_1.ipynb) · [Assignment](mfml/lab-1/Lab_1.pdf) · [Joint report: labs 1–2](mfml/lab-1/MOMO_TEAM7_LAB1_LAB2.pdf) |
| 2 · Numerical optimization | Compare BFGS, L-BFGS, and Adam on a quadratic objective and logistic regression, including convergence behavior. | [Notebook](mfml/lab-2/MFML_Team_7_LAB_2.ipynb) · [Assignment](mfml/lab-2/Lab_2.pdf) · [Joint report: labs 1–2](mfml/lab-2/MOMO_TEAM7_LAB1_LAB2.pdf) |
| 3 · Medical appointment attendance | Analyze appointment no-shows, handle data quality issues, and compare predictive models. | [Notebook](mfml/lab-3/MFML_Team_7_Lab_3.ipynb) · [Assignment](mfml/lab-3/Lab_3.pdf) · [Report](mfml/lab-3/MFML_Team_7_Lab_3.pdf) |
| 4 · Multivariate and Bayesian analysis | Explore PCA, t-SNE, UMAP, factor analysis, clustering, and Bayesian networks on appointment data. | [Notebook](mfml/lab-4/MFML_Team_7_Lab_4.ipynb) · [Assignment](mfml/lab-4/Lab_4.pdf) · [Report](mfml/lab-4/MFML_Team_7_Lab_4.pdf) · [Network exports](mfml/lab-4/) |

The [combined labs 3–4 report](mfml/lab-4/MFML_TEAM7_LAB3_LAB4.pdf) brings the latter two studies together.

## Deep Learning

Controlled experiments on neural architectures, training choices, and evaluation. The labs cover image classification, conversational language models, agent orchestration for scientific review, and latent-variable generative models.

| Lab | What it is about | Artifacts |
| --- | --- | --- |
| 1 · CNN image classification | Compare preprocessing, augmentation, architectures, optimizers, and schedules on CIFAR-10, Fashion-MNIST, and SVHN. | [Notebook](dl/lab-1/334771_lab_1_cnn.ipynb) · [Assignment](dl/lab-1/Lab_1.pdf) · [Requirements](dl/lab-1/requirements.txt) · [Results cache](dl/lab-1/results_cache.zip) |
| 2 · Language model fine-tuning | Prepare OASST1 conversations, fine-tune a causal transformer, evaluate perplexity/BLEU/ROUGE, and examine decoding choices. | [Notebook](dl/lab-2/334771_lab_2_llm.ipynb) · [Assignment](dl/lab-2/Lab_2.pdf) · [Requirements](dl/lab-2/requirements.txt) |
| 3 · Agentic scientific review | Compare a baseline review pipeline with supervisor/planner/worker orchestration through controlled experiments, ablations, and evaluation. | [Notebook](dl/lab-3/334771_lab_3_mas.ipynb) · [Assignment](dl/lab-3/Lab_3.pdf) · [Requirements](dl/lab-3/requirements.txt) · [Results cache](dl/lab-3/results_cache.zip) |
| 4 · Autoencoders and generation | Compare PCA, AE, VAE, and GAN on MNIST; study latent spaces, generation, reconstruction, and denoising. | [Notebook](dl/lab-4/334771_lab_4_vae.ipynb) · [Assignment](dl/lab-4/Lab_4.pdf) · [Requirements](dl/lab-4/requirements.txt) |

## Machine Learning for Time Series

A connected series of studies using electricity load, normal ECG examples, and SPY prices. It progresses from describing temporal structure to comparing forecasting methods and investigating state-space representations. The causal-analysis lab also includes DTW alignment of MIT-BIH ECG records. Later labs study coupled oscillator networks and Hawkes models for large-return events.

| Lab | What it is about | Artifacts |
| --- | --- | --- |
| 1 · Time series characterization | Use decomposition, FFT, autocorrelation, Hurst and Lyapunov estimates, and entropy measures to discuss predictability. | [Notebook](tsml/lab-1/M1_Lopatenko.ipynb) · [Assignment](tsml/lab-1/lab1.pdf) · [Figures](tsml/lab-1/figures/) |
| 2 · Classical forecasting | Tune and compare classical forecasting methods, including custom versus framework implementations and repeated out-of-sample evaluation. | [Notebook](tsml/lab-2/M2_Lopatenko.ipynb) · [Assignment](tsml/lab-2/lab2.pdf) · [Results](tsml/lab-2/forecast_mape_summary.csv) |
| 3 · Causal analysis and DTW | Examine cross-correlation, Granger tests, and convergent cross mapping; compare ECG sequences with dynamic time warping. | [Notebook](tsml/lab-3/M3_Lopatenko.ipynb) · [Assignment](tsml/lab-3/lab3.pdf) |
| 4 · Neural forecasting | Compare perceptrons and LSTM with classical methods over matched forecast horizons and test blocks, including training cost. | [Notebook](tsml/lab-4/M4_Lopatenko.ipynb) · [Assignment](tsml/lab-4/lab4.pdf) |
| 5 · Reconstructed dynamics | Estimate delay and embedding dimension, visualize trajectories, build state-transition graphs, and compare reconstructed inputs for LSTM forecasting. | [Notebook](tsml/lab-5/M5_Lopatenko.ipynb) · [Assignment](tsml/lab-5/lab5.pdf) |
| 6 · Local, analogue, and SSA forecasting | Compare local approximation, maximum-similarity forecasting, and singular spectrum analysis, relating performance to the structure of each series. | [Notebook](tsml/lab-6/M6_Lopatenko.ipynb) · [Assignment](tsml/lab-6/lab6.pdf) |
| 7 · Coupled oscillator forecasting | Train a damped oscillator network and compare forecast errors and median CPU runtimes with MLP and LSTM baselines on matched test blocks. | [Notebook](tsml/lab-7/M7_Lopatenko.ipynb) · [Assignment](tsml/lab-7/lab7-8.pdf) |
| 8 · Hawkes event modeling | Model large SPY return events with Hawkes and Poisson processes; compare event intervals, magnitudes, and counts using likelihood, Wasserstein distance, and KL divergence. | [Notebook](tsml/lab-8/M8_Lopatenko.ipynb) · [Assignment](tsml/lab-8/lab7-8.pdf) |

## Automated Machine Learning

Experiments on how to choose model configurations and learning algorithms automatically. The emphasis is on explicit search spaces, comparable evaluation budgets, held-out evaluation, and the computational cost of model selection.

| Lab | What it is about | Artifacts |
| --- | --- | --- |
| 1 · Bayesian optimization | Compare a manual Gaussian-process/Expected Improvement optimizer, random search, and Optuna samplers for gradient boosting on MAGIC Gamma Telescope. | [Notebook](automl/lab-1/334771_Lopatenko_bayesian_opt.ipynb) · [Assignment](automl/lab-1/1.%20Bayesian%20optimization.md) · [Requirements](automl/lab-1/reqs.txt) |
| 2 · Meta-features and algorithm selection | Compare hand-designed and PyMFE dataset descriptions for predicting the best algorithm in a fixed portfolio; examine invariance and meta-model performance. | [Notebook](automl/lab-2/334771_Lopatenko_meta_features.ipynb) · [Assignment](automl/lab-2/2.%20Meta-features.md) · [Experiment runner](automl/lab-2/run_meta_experiments.py) · [Requirements](automl/lab-2/reqs.txt) |
