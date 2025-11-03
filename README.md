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

Gradient-based models in general
