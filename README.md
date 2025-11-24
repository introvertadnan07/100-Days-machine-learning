# 100-Days-machine-learning

A structured 100-day learning project focused on machine learning fundamentals, executed via Jupyter notebooks.

🧭 Overview

This repository documents a day-by-day learning journey in machine learning. Each day’s notebook walks through a new concept, exercise, or small project—building step by step from data handling into modelling and analysis.
The aim is to:

Build a strong foundation in ML concepts.

Gain hands-on experience with data in notebook form.

Track progress over a sustained time period (100 days).

Create a portfolio of notebooks you can revisit and reuse.

 day - 16
 
Normalization in Machine Learning

This notebook explains how normalization works and why it matters when working with machine learning models. Normalization helps bring different features to a similar scale, which makes model training more stable and often improves performance.

What This Notebook Covers

What normalization means and when we use it

The difference between normalization and standardization

How scaling affects model performance

Using MinMaxScaler with real examples

Visual comparison before and after normalization

Why Normalization Is Important

Some algorithms rely on distance calculations or gradient updates. If one feature has a larger range than others, it can dominate the results. Normalization solves this by scaling values to a fixed range, usually between 0 and 1.

Models that benefit the most:

K-Nearest Neighbors

K-Means clustering

Neural networks

Linear and Logistic Regression (in many cases)

17 | Encoding

This notebook is part of the 100 Days of Machine Learning series. It covers the topic of categorical and numerical encoding techniques, showing how to prepare non-numeric features for machine learning models.

Table of Contents

Motivation

Objectives

Dataset overview

Encoding techniques

Label encoding

One-hot encoding

Ordinal encoding

Frequency / target encoding (if covered)

Implementation (with code)

Model training & evaluation (post-encoding)

Key takeaways

Further reading & references

Motivation

Many machine learning algorithms expect numeric input. Real-world datasets often contain categorical (string/object) features such as colors, labels, or categories. Encoding converts these into a numeric form while preserving useful information. This notebook demonstrates how different encoding methods affect model performance and feature representation.

Objectives

By the end of this notebook you will be able to:

Identify types of categorical features (nominal, ordinal).

Choose an appropriate encoding method for a given categorical feature.

Implement encoding using pandas, scikit-learn, or other relevant libraries.

Understand how encoding interacts with downstream modelling (e.g., tree-based vs linear models).

Recognize pitfalls: high-cardinality features, dummy-variable trap, information leakage.

Dataset overview

In this notebook:

We load a sample dataset (describe the name of the dataset, number of rows/columns, and main categorical features).

We highlight the categorical features that require encoding and examine their distributions.

We split into training/test (or use cross-validation) ensuring encoding is applied correctly (i.e., fit only on training data, transform on test).

Encoding techniques
Label encoding

Convert each category value to an integer code (e.g., Red→0, Green→1, Blue→2).

Suitable for ordinal features (where categories have a meaningful order).

Risky with nominal features if model interprets numeric codes as ordered.

One-hot encoding

Create binary/dummy variables for each category (e.g., Color_Red, Color_Green, Color_Blue).

Works well for nominal features with relatively low cardinality.

Can lead to high dimensionality if many categories.

Beware the dummy-variable trap in linear models (perfect multicollinearity).

Ordinal encoding

Map categories to integer codes according to a defined order (e.g., Low→1, Medium→2, High→3).

Only makes sense when the ordering is meaningful.

Must define the ordering consciously.

(Optional) Frequency or target encoding

Frequency encoding replaces categories by their count or proportion in the dataset.

Target encoding replaces categories by the average target value for that category.

Useful for high-cardinality categorical features.

Must handle leakage and overfitting with care (use CV, smoothing, etc.).

Implementation

Use pandas to explore and inspect categorical features: .dtypes, .value_counts().

Use scikit-learn’s LabelEncoder, OneHotEncoder, OrdinalEncoder where suitable.

Show before/after transformation shapes, sample transformed data.

Illustrate how encoded data integrates with model input pipelines (e.g., ColumnTransformer).

Show how to handle unseen categories in test set (set handle_unknown='ignore').

Model training & evaluation

Use a simple model (for example: logistic regression, random forest) to compare performance changes with different encoding techniques.

Evaluate metrics (accuracy, F1 score, etc.) and observe how encoding choices impact them.

Discuss model interpretability implications: coefficients for encoded variables, feature importance.

Key takeaways

There is no one-size-fits-all encoding method. The right choice depends on: the type of feature (nominal vs ordinal), model type, number of categories.

Properly fitting encoding only on training data is critical to avoid data leakage.

High-cardinality categorical features need special handling (e.g., dimensionality reduction, encoding with smoothing).

Encoding affects both model performance and interpretability—keep feature engineering consistent and transparent.

Always test different encoding strategies when building end-to-end pipelines.

Further reading & references

“Feature Engineering for Machine Learning” by Alice Zheng & Amanda Casari

scikit-learn documentation: Preprocessing categorical features

Articles on target encoding and leakage avoidance

19 – Column Transformer

Part of the 100-Days-Machine-Learning series by introvertadnan07

📄 Overview

In this notebook, we dive into the use of the ColumnTransformer class (from libraries like scikit-learn) to build preprocessing pipelines that apply different transformations to different columns – e.g., numeric vs categorical features.
You’ll see how to:

separate feature types (numerical, categorical)

apply standard scaling, one-hot encoding (or other encodings)

integrate preprocessing into a full ML pipeline

improve code clarity and maintainability with transformers

✅ Why this matters

Pre-processing is often one of the most error-prone and messy parts of a machine-learning workflow. Using a ColumnTransformer helps by:

keeping transformations organised

ensuring consistent handling of train/test data

streamlining pipelines for models and deployment

🧠 What you’ll learn

By working through this notebook you should become comfortable with:

identifying columns by data type or role in the dataset

using ColumnTransformer with Pipeline to chain steps

comparing performance and maintenance benefits compared to ad-hoc preprocessing

writing cleaner code that integrates effortlessly into training and inference

20 – Pipelines

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

This notebook introduces and demonstrates the use of pipelines (for example using scikit‑learn’s Pipeline, make_pipeline, and related tools) to streamline full data-preprocessing + model training flows. By chaining steps together, you’ll learn to build clean, repeatable, and maintainable ML workflows.

🎯 What you’ll cover

Crafting pipelines that unify preprocessing (scaling, encoding, etc) and model training

Seeing how pipelines ease handling of train/test splits, cross-validation, and deployment

Understanding how pipelines help avoid data-leakage and improve code modularity

Enhancing readability and maintainability of ML scripts or notebooks

🔧 Why it matters

As ML workflows grow in complexity (multiple feature types, transforms, models, evaluation steps), using plain ad-hoc code gets error-prone and hard to manage. Pipelines bring structure, enforce the correct order of operations, and make it easier to update or replace one step without breaking the rest.

Gradient-based models in general

21 – Without Pipelines

Part of the 100-Days-Machine-Learning series by introvertadnan07

📘 Overview

This notebook walks through how to perform preprocessing, model training, and evaluation without using pipelines. It serves as a practical comparison to the previous notebook — 20 – Pipelines — helping you understand why pipelines are valuable and what problems they solve.

By working manually through each step, you’ll see the inner workings of data preparation and how each component fits together before automation.

🎯 Learning objectives

Understand how to handle preprocessing manually (scaling, encoding, transformation)

Learn how to fit transformers and models separately

Observe how data leakage can occur if steps are not properly isolated

Compare the workflow’s complexity and maintainability with the pipeline approach

🧠 Key takeaways

Manual steps = flexibility + higher risk. Doing each transformation by hand gives more control but increases the chance of mistakes.

Pipelines = consistency. This notebook highlights why scikit-learn pipelines simplify the process.

22 – Titanic using Pipeline

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

This notebook applies a full pipeline workflow on the classic Titanic dataset. You’ll build preprocessing, encoding, feature-engineering and model training steps as a unified flow using tools like Pipeline, ColumnTransformer, and a classification algorithm. It shows how to move from raw CSV to a clean model in a structured, repeatable way.

🎯 What you’ll cover

Loading and preparing the Titanic dataset (features such as Pclass, Sex, Age, etc.)

Handling missing values and feature engineering (e.g., combining SibSp + Parch, extracting titles)

Building separate transformations for numerical and categorical data

Integrating those transformations into a ColumnTransformer, then wrapping in a Pipeline along with model training

Evaluating the model’s performance (accuracy, confusion matrix, etc) and interpreting results

🧠 Why it matters

Working through this notebook helps you understand:

Why pipelines improve reproducibility and reduce errors compared to ad-hoc scripts

How to structure code so preprocessing and model training are clearly separated and chained

Best practices for real-world data workflows (feature selection, data leakage prevention, modularization)

📂 How it fits in

This is Day 22 of your series.
It builds on previous days—which introduced pipelines and manual workflows—by taking a tangible dataset (Titanic) and applying a full end-to-end pipeline. It sets the foundation for even more complex workflows (hyperparameter tuning, deployment) in subsequent days.

Transparency matters. Before automating, understanding the full process helps you debug and design better ML systems.

23 – Titanic Model Pipeline

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

This notebook takes the classic Titanic disaster dataset and builds a full, streamlined machine-learning pipeline: from data preparation, through feature engineering, all the way to model training and evaluation. The goal is to demonstrate how to assemble and manage a robust workflow for real-world ML problems.

🎯 What you’ll cover

Loading and cleaning the Titanic dataset

Feature engineering: creating new features, encoding categorical variables, imputing missing values

Constructing a preprocessing pipeline (handling numeric and categorical features separately)

Building a complete pipeline that includes preprocessing + model training (e.g., classification algorithm)

Evaluating model performance and understanding how each component contributes to the final outcome

Inspecting and saving the pipeline for future reuse or deployment

🧠 Why it matters

Pipelines increase reproducibility and reduce risk of errors when deploying ML workflows.

They ensure transformations applied during training are exactly the same during inference.

This notebook connects the theory of pipelines with a concrete dataset and shows how to manage feature engineering + modelling in a clean, maintainable manner.

24 – Function Transformer

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

In this notebook you’ll explore how to use the FunctionTransformer (from scikit‑learn) to wrap custom functions into your preprocessing pipelines. This lets you apply bespoke transformations (e.g., custom feature engineering, mathematical operations) in the same consistent way you apply built-in transformers.

scikit-learn.org
+1

🎯 What you’ll cover

Defining custom functions for data transformation (for example log-scaling, combining columns, feature extraction)

Wrapping those functions into a FunctionTransformer so they behave like any other transformer (with fit, transform, and compatibility with pipelines)

Incorporating the FunctionTransformer into a full pipeline (possibly with other preprocessing steps + model)

Understanding the nuances: when to use custom functions vs built-in transformers, and what to watch out for (e.g., picklability when using lambda functions) 
scikit-learn.org
+1

Experimenting with the pipeline and seeing how your custom transform affects model performance or data flow

🧠 Why it matters

Flexibility: Built-in transformers cover many use-cases but not every unique manipulation you’ll need.

Consistency: By wrapping custom transforms you keep your workflows pipeline-friendly, reusable, and fit for cross-validation or production.

Maintainability: Everything plays nicely together in one pipeline structure rather than ad-hoc code scattered around.

Risk control: Leveraging custom functions inside pipelines reduces the chance of data-leakage, ensures you apply operations uniformly across train/test splits.

26 – Power Transform

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

In this notebook you explore the technique of using power-transformations (such as PowerTransformer in scikit‑learn) to make numerical features more “Gaussian-like”. This is a key preprocessing step when you have skewed data or variables whose distributions violate assumptions of some machine-learning models. 
Scikit-learn
+1

🎯 What you’ll cover

Identifying numeric features with skew or non-normal distributions

Applying the PowerTransformer (methods like “yeo-johnson” or “box-cox”) to stabilize variance and reduce skewness. 
Scikit-learn

Integrating a power transform into a preprocessing pipeline (e.g., combining with scaling/encoding)

Visualising before/after distributions to measure the impact of transformation

Observing how transformed data affects model training and evaluation

Understanding when to use power transforms and when they may not offer benefit

🧠 Why it matters

Many models (especially linear ones) expect features to be roughly Gaussian or to have stable variance. Power transforms help bring distributions closer to that ideal. 
MachineLearningMastery.com

Reducing skew improves model robustness and interpretability of features.

Building pipeline-friendly transformations ensures the same steps apply in train/test, avoiding data leakage.

📂 How it fits in the series

This is Day 26 of your series.

27 – Binning and Binarization

Part of the 100-Days-Machine-Learning series by introvertadnan07

📖 Purpose

This notebook introduces two important preprocessing techniques: binning (also known as discretization) and binarization (turning continuous or categorical features into binary features). These methods help when you want to simplify feature distributions or convert into a format many algorithms or models prefer.

🎯 What you’ll cover

Detecting features that may benefit from discretization (e.g., continuous numeric data with skew or wide ranges)

Applying binning methods (fixed bins, quantile bins, custom thresholds)

Converting features into binary form: thresholds, one-hot style, presence/absence flags

Working with built-in transformers from libraries like scikit‑learn (e.g., KBinsDiscretizer, Binarizer)

Integrating these transforms into a preprocessing pipeline and observing how they affect model behaviour

Visualising before/after results: how distributions change, how discretized/binarised features behave

Discussing when you shouldn’t use binning or binarization (loss of information, risk of arbitrary thresholds)

🧠 Why it matters

Some models or algorithms handle discrete/binary features more naturally or efficiently than highly continuous ones.

Binning can reduce sensitivity to outliers or non-linear relationships by categorizing ranges.

Binarization makes features more interpretable (e.g., “age > 40” flag) and often easier to include in simple modelling strategies.

Ensuring these transformations are inside pipelines preserves consistency across train/test splits and avoids leakage.

📂 How it fits in the series

This is Day 27 of your series. Up to now you’ve covered scaling, encoding, power transforms, custom transforms etc. Binning and binarization build on that — now we’re dealing with ways to reshape feature distributions and format features for downstream algorithms. This sets a strong base for feature engineering and modelling days ahead.
It builds upon earlier preprocessing work (e.g., encoding, scaling, custom transforms) and adds one more powerful tool to your workflow. After this, you’re in a stronger position to build robust ML pipelines that handle real-world data quirks.

