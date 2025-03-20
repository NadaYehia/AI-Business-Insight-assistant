# AI-Business-Insight-assistant

This project provides a comprehensive analysis of sales data using Python and integrates an AI agent for interactive recommendations and insights. The code is divided into several sections, including data preprocessing, advanced data summarization, visualization, and an AI-powered agent for answering user queries.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Code Overview](#code-overview)
3. [Data Analysis](#data-analysis)
4. [Visualization](#visualization)
5. [AI Agent](#ai-agent)
6. [Model Evaluation](#model-evaluation)
7. [Streamlit App](#streamlit-app)

---

## Introduction
This project analyzes sales data to extract key insights, such as total sales, best/worst performing months, and top-selling products. It also includes an AI agent powered by OpenAI's GPT-3.5-turbo model to provide recommendations and answer user queries based on the analyzed data.

---
## How to run the code
```python


pip install streamlit
streamlit run BI_insight_companion.py

```

## Code Overview
The code is structured as follows:
1. **Data Loading and Preprocessing**: Load the sales data and preprocess it for analysis.
2. **Advanced Data Summary**: Compute key metrics such as total sales, average sales, and best/worst performing months.
3. **Visualization**: Generate visualizations of sales trends by product.
4. **AI Agent**: Create an AI agent using LangChain and OpenAI to provide recommendations and answer questions.
5. **Model Evaluation**: Evaluate the AI agent's performance using predefined question-answer pairs.
6. **Streamlit App**: Build an interactive web app using Streamlit for data visualization and AI agent interaction.

---

## Data Analysis
### Key Metrics
- **Total Sales**: Sum of all sales.
- **Average Sales**: Mean sales value.
- **Standard Deviation of Sales**: Measure of sales variability.
- **Best/Worst Month**: Month with the highest/lowest sales.
- **Top-Selling Product**: Product with the highest total sales.
- **Most Frequent Product**: Product with the highest number of sales transactions.

## Visualization
The plt_sales_by_product function generates a plot of smoothed sales data for each product over time.

## Model Evaluation
The AI agent is evaluated using predefined question-answer pairs to ensure accuracy.

## AI Agent
The AI agent is built using LangChain and OpenAI's GPT-3.5-turbo model. It provides recommendations and answers user queries based on the sales data summary.

Tools
OverallChainTool: Provides recommendations based on the sales data summary.

GeneralChatTool: Handles unrelated enquiries.

RecallHistoryTool: Recalls chat history if requested.


## Streamlit App
The Streamlit app provides an interactive interface for data visualization and AI agent interaction.


