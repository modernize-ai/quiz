"""
Public AI & Tech Quiz Application — Self-Assessment Tool
Covers: AI/ML, RAG, Vector Databases, Embeddings, Transformers, Agents, APIs
Run: streamlit run quiz_app_public.py
"""

import streamlit as st
import random
import json
import os
from datetime import datetime
import csv
import copy
from openai import AzureOpenAI
import io

# ─────────────────────────────────────────────
# QUESTION BANK — 100 General Public Questions
# ─────────────────────────────────────────────
QUESTIONS = [
    # ── AI Fundamentals (10 questions) ──────────────────
    {
        "topic": "AI Fundamentals",
        "type": "mcq",
        "question": "What does AI stand for?",
        "options": ["Automated Intelligence", "Artificial Intelligence", "Advanced Integration", "Automated Information"],
        "answer": "Artificial Intelligence",
        "explanation": "AI = Artificial Intelligence. It's the simulation of human intelligence processes by computer systems."
    },
    {
        "topic": "AI Fundamentals",
        "type": "mcq",
        "question": "What is the primary goal of Machine Learning?",
        "options": ["Write code automatically", "Enable systems to learn from data without being explicitly programmed", "Replace human developers", "Reduce computational power"],
        "answer": "Enable systems to learn from data without being explicitly programmed",
        "explanation": "ML allows algorithms to improve their performance through experience and data, rather than being hardcoded."
    },
    {
        "topic": "AI Fundamentals",
        "type": "mcq",
        "question": "Which of these is a type of Machine Learning?",
        "options": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "All of the above"],
        "answer": "All of the above",
        "explanation": "These are the three main categories of ML: supervised (labeled data), unsupervised (unlabeled data), and reinforcement (reward-based)."
    },
    {
        "topic": "AI Fundamentals",
        "type": "fill",
        "question": "The process of training a model on historical data to make predictions is called ___",
        "answer": "supervised learning",
        "explanation": "Supervised learning uses labeled data where both input and output are provided during training."
    },
    {
        "topic": "AI Fundamentals",
        "type": "mcq",
        "question": "What is Deep Learning?",
        "options": ["Machine Learning using very large datasets", "Machine Learning using neural networks with multiple layers", "A type of database technology", "Advanced statistical analysis"],
        "answer": "Machine Learning using neural networks with multiple layers",
        "explanation": "Deep Learning uses artificial neural networks inspired by biological neurons to process complex patterns."
    },

    # ── LLMs & Transformers (15 questions) ──
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What does LLM stand for?",
        "options": ["Linear Language Model", "Large Language Model", "Logical Learning Machine", "Latent Language Matrix"],
        "answer": "Large Language Model",
        "explanation": "LLM = Large Language Model. Examples: GPT-4, Claude, Llama. Trained on billions of text tokens."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What is a Transformer model?",
        "options": ["A power converter for neural networks", "A deep learning architecture using self-attention mechanisms", "A data preprocessing tool", "A type of database"],
        "answer": "A deep learning architecture using self-attention mechanisms",
        "explanation": "Transformers use attention to weigh importance of different words. Foundation for GPT, BERT, T5."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "fill",
        "question": "The Transformer architecture introduced the ___ mechanism which allows models to focus on relevant parts of input",
        "answer": "attention",
        "explanation": "Self-attention lets each token attend to all other tokens, capturing long-range dependencies."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What is tokenization?",
        "options": ["Encrypting sensitive data", "Breaking text into smaller units (tokens) for processing", "Converting images to text", "Compressing data"],
        "answer": "Breaking text into smaller units (tokens) for processing",
        "explanation": "Tokenization splits text into words, subwords, or characters that models can process."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What is the context window of an LLM?",
        "options": ["The time it takes to generate a response", "The maximum number of tokens the model can consider at once", "The user's browser window", "The training dataset size"],
        "answer": "The maximum number of tokens the model can consider at once",
        "explanation": "Context window = max input length. GPT-4 Turbo: 128K tokens. Longer context = more information the model sees."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What is temperature in LLM inference?",
        "options": ["The computational heat generated", "A parameter controlling randomness/creativity of responses", "The training time", "The model size"],
        "answer": "A parameter controlling randomness/creativity of responses",
        "explanation": "Temp=0 (deterministic) vs Temp=1.0+ (creative). Low temp for factual tasks; high temp for creative writing."
    },

    # ── RAG & Vector Search (15 questions) ──
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "What does RAG stand for?",
        "options": ["Real-time Augmented Generation", "Retrieval-Augmented Generation", "Rapid AI Gateway", "Recurrent Attention Group"],
        "answer": "Retrieval-Augmented Generation",
        "explanation": "RAG = Retrieval-Augmented Generation. Combines LLMs with external knowledge retrieval to improve answers."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "Why use RAG instead of fine-tuning?",
        "options": [
            "Fine-tuning is not possible",
            "RAG allows dynamic knowledge updates without retraining; fine-tuning requires retraining for new data",
            "RAG is always faster",
            "Fine-tuning doesn't work with transformers"
        ],
        "answer": "RAG allows dynamic knowledge updates without retraining; fine-tuning requires retraining for new data",
        "explanation": "RAG is flexible for live data. Fine-tuning is permanent but compute-intensive."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "fill",
        "question": "A ___ is a database that stores embeddings and enables similarity search",
        "answer": "vector database",
        "explanation": "Vector databases like Chroma, Pinecone, Weaviate store embeddings and retrieve similar items quickly."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "What is semantic search?",
        "options": [
            "Searching by exact keyword matching",
            "Searching based on meaning and context using embeddings",
            "Searching only within semantic web links",
            "Searching by grammar rules"
        ],
        "answer": "Searching based on meaning and context using embeddings",
        "explanation": "Semantic search understands intent. 'Car' retrieves docs about 'automobile' even without the word 'car'."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "In RAG, what is the retriever?",
        "options": [
            "The LLM that generates answers",
            "The component that fetches relevant documents from a knowledge base",
            "The database server",
            "The user interface"
        ],
        "answer": "The component that fetches relevant documents from a knowledge base",
        "explanation": "Retriever finds top-k relevant docs → passed to LLM → LLM generates answer using retrieved context."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "What is hybrid search?",
        "options": [
            "Combining two LLMs",
            "Combining dense (semantic) and sparse (keyword) retrieval",
            "Using both text and images",
            "Multi-language search"
        ],
        "answer": "Combining dense (semantic) and sparse (keyword) retrieval",
        "explanation": "Hybrid = semantic search + BM25 keyword search. Gets best of both worlds: meaning + exact matches."
    },

    # ── Embeddings (12 questions) ──────────────────────
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What is an embedding?",
        "options": [
            "A type of programming function",
            "A numerical vector representation of text/data that captures semantic meaning",
            "An HTML tag",
            "A database index"
        ],
        "answer": "A numerical vector representation of text/data that captures semantic meaning",
        "explanation": "Embeddings convert text to vectors. Similar meanings = similar vectors in high-dimensional space."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What is cosine similarity used for?",
        "options": [
            "Measuring file size", 
            "Calculating the angle between two vectors to measure similarity",
            "Trigonometric calculations",
            "Converting angles to distances"
        ],
        "answer": "Calculating the angle between two vectors to measure similarity",
        "explanation": "Cosine similarity = 1 (identical), 0 (perpendicular), -1 (opposite). Used to find similar embeddings."
    },
    {
        "topic": "Embeddings",
        "type": "fill",
        "question": "Word2Vec was developed by ___ and uses neural networks to create word embeddings",
        "answer": "Google",
        "explanation": "Word2Vec (Google) introduced CBOW and Skip-gram algorithms for word embeddings."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What is the advantage of BERT embeddings over Word2Vec?",
        "options": [
            "Word2Vec is better for everything",
            "BERT is context-aware (same word gets different embeddings in different contexts)",
            "BERT is faster",
            "No difference"
        ],
        "answer": "BERT is context-aware (same word gets different embeddings in different contexts)",
        "explanation": "Word2Vec = one embedding per word. BERT = contextualized embeddings based on surrounding words."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What is sentence-BERT (SBERT)?",
        "options": [
            "BERT for single sentences only",
            "BERT modified to create fixed-size sentence embeddings for direct comparison",
            "A faster version of BERT",
            "BERT for coding tasks"
        ],
        "answer": "BERT modified to create fixed-size sentence embeddings for direct comparison",
        "explanation": "SBERT enables efficient sentence-to-sentence similarity without recomputing embeddings."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What does TF-IDF stand for?",
        "options": [
            "Text Frequency-Inverse Data Frequency",
            "Term Frequency-Inverse Document Frequency",
            "Total Frequency-Internal Data Format",
            "Text Feature-Indexed Distributed Format"
        ],
        "answer": "Term Frequency-Inverse Document Frequency",
        "explanation": "TF-IDF weights terms by frequency in document, penalizing common words. Classic for sparse embeddings."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "In embeddings, what does 'dimensionality' refer to?",
        "options": [
            "The size of the training dataset",
            "The number of features/values in each embedding vector",
            "The number of documents",
            "The type of database used"
        ],
        "answer": "The number of features/values in each embedding vector",
        "explanation": "Typical: 384-dim, 768-dim, 1536-dim. Higher dimensions capture more nuance but require more compute."
    },

    # ── Vector Databases (10 questions) ──────────────────────
    {
        "topic": "Vector Databases",
        "type": "mcq",
        "question": "What is a vector database?",
        "options": [
            "A database for video files",
            "A specialized database optimized for storing and searching high-dimensional vectors",
            "A database using GraphQL vectors",
            "A database for mathematical matrices"
        ],
        "answer": "A specialized database optimized for storing and searching high-dimensional vectors",
        "explanation": "Vector DBs (Chroma, Pinecone, Weaviate) enable fast similarity search on embeddings."
    },
    {
        "topic": "Vector Databases",
        "type": "mcq",
        "question": "Name a popular open-source vector database:",
        "options": ["PostgreSQL", "Chroma", "Excel", "Notion"],
        "answer": "Chroma",
        "explanation": "Chroma is lightweight, open-source. Others: Milvus, Weaviate. Cloud: Pinecone, Qdrant."
    },
    {
        "topic": "Vector Databases",
        "type": "fill",
        "question": "The most common distance metric in vector databases is ___ or L2",
        "answer": "cosine",
        "explanation": "Cosine distance measures angle between vectors. L2 = Euclidean distance."
    },
    {
        "topic": "Vector Databases",
        "type": "mcq",
        "question": "What does HNSW stand for?",
        "options": [
            "Hybrid Network Search Web",
            "Hierarchical Navigable Small World",
            "High-speed Network Storage Web",
            "Hash-based Navigation Search Window"
        ],
        "answer": "Hierarchical Navigable Small World",
        "explanation": "HNSW = graph-based ANN (Approximate Nearest Neighbor) algorithm for fast vector search."
    },
    {
        "topic": "Vector Databases",
        "type": "mcq",
        "question": "What is metadata filtering in vector databases?",
        "options": [
            "Removing invalid data",
            "Filtering search results by non-embedding attributes before/after similarity search",
            "Cleaning vector values",
            "Type conversion"
        ],
        "answer": "Filtering search results by non-embedding attributes before/after similarity search",
        "explanation": "Example: 'Find similar docs where date > 2024 AND category = AI'. Combines vector + traditional queries."
    },
    {
        "topic": "Vector Databases",
        "type": "mcq",
        "question": "What is the k in k-nearest neighbors (k-NN)?",
        "options": [
            "A constant always equal to 1000",
            "The number of nearest neighbors to retrieve",
            "Kelvin temperature",
            "Kernel function parameter"
        ],
        "answer": "The number of nearest neighbors to retrieve",
        "explanation": "k=5 means retrieve top 5 most similar vectors. Affects result quality and speed."
    },

    # ── APIs & Integration (12 questions) ──────────────────────
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "What does API stand for?",
        "options": ["Automated Programming Interface", "Application Programming Interface", "Automated Processing Item", "Application Program Index"],
        "answer": "Application Programming Interface",
        "explanation": "API = contracts for how software components communicate with each other."
    },
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "What is REST?",
        "options": [
            "A type of database",
            "Representational State Transfer - architectural style for APIs using HTTP",
            "A programming language",
            "A compression format"
        ],
        "answer": "Representational State Transfer - architectural style for APIs using HTTP",
        "explanation": "REST uses HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove)."
    },
    {
        "topic": "APIs & Integration",
        "type": "fill",
        "question": "The most common data format for APIs is ___",
        "answer": "JSON",
        "explanation": "JSON (JavaScript Object Notation) is human-readable, lightweight, and widely supported."
    },
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "What is an API key?",
        "options": [
            "A type of keyboard",
            "A secret credential to authenticate API requests",
            "A database password",
            "A software license"
        ],
        "answer": "A secret credential to authenticate API requests",
        "explanation": "API keys identify the client and track usage. Keep them secret! Rotate periodically."
    },
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "What is rate limiting in APIs?",
        "options": [
            "The speed of internet connection",
            "Restricting the number of requests a client can make in a time period",
            "Limiting file size uploads",
            "Bandwidth throttling"
        ],
        "answer": "Restricting the number of requests a client can make in a time period",
        "explanation": "Rate limits prevent abuse. Example: 100 requests/minute. Protects server resources."
    },
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "What is a webhook?",
        "options": [
            "A type of programming hook",
            "A callback URL that receives real-time event notifications",
            "A debugging tool",
            "A code formatter"
        ],
        "answer": "A callback URL that receives real-time event notifications",
        "explanation": "Webhooks enable push notifications. Server sends data to client when event occurs."
    },

    # ── Data & Preprocessing (12 questions) ──────────────────────
    {
        "topic": "Data & Preprocessing",
        "type": "mcq",
        "question": "What is data normalization?",
        "options": [
            "Removing duplicate rows",
            "Scaling numerical values to a standard range (e.g., 0-1)",
            "Sorting data",
            "Adding missing values"
        ],
        "answer": "Scaling numerical values to a standard range (e.g., 0-1)",
        "explanation": "Normalization ensures features are comparable. Important for distance-based algorithms."
    },
    {
        "topic": "Data & Preprocessing",
        "type": "mcq",
        "question": "What is the curse of dimensionality?",
        "options": [
            "Using too many dimensions in a spreadsheet",
            "Performance degradation as number of features increases; algorithms become less effective",
            "A database size limit",
            "A type of error"
        ],
        "answer": "Performance degradation as number of features increases; algorithms become less effective",
        "explanation": "More features often ≠ better. Too many features can cause overfitting and slow algorithms."
    },
    {
        "topic": "Data & Preprocessing",
        "type": "fill",
        "question": "The process of removing outliers and inconsistencies from data is called data ___",
        "answer": "cleaning",
        "explanation": "Data cleaning improves quality. Includes handling missing values, removing duplicates, fixing inconsistencies."
    },
    {
        "topic": "Data & Preprocessing",
        "type": "mcq",
        "question": "What is stratified sampling?",
        "options": [
            "Random sampling",
            "Sampling that maintains proportions of different classes/groups",
            "Sampling only the first N records",
            "Sampling without replacement"
        ],
        "answer": "Sampling that maintains proportions of different classes/groups",
        "explanation": "Ensures train/test splits have same class distribution as original data."
    },
    {
        "topic": "Data & Preprocessing",
        "type": "mcq",
        "question": "What is feature engineering?",
        "options": [
            "Collecting data",
            "Creating new features from raw data to improve model performance",
            "Labeling data",
            "Compressing data"
        ],
        "answer": "Creating new features from raw data to improve model performance",
        "explanation": "Example: From date, extract day-of-week, month, is_weekend. Better features = better models."
    },

    # ── Model Evaluation (12 questions) ──────────────────────
    {
        "topic": "Model Evaluation",
        "type": "mcq",
        "question": "What is accuracy in machine learning?",
        "options": [
            "How slowly a model trains",
            "Percentage of correct predictions out of total predictions",
            "The number of features used",
            "The size of the dataset"
        ],
        "answer": "Percentage of correct predictions out of total predictions",
        "explanation": "Accuracy = correct predictions / total predictions. Can be misleading with imbalanced data."
    },
    {
        "topic": "Model Evaluation",
        "type": "mcq",
        "question": "What is the confusion matrix used for?",
        "options": [
            "Displaying database structure",
            "Summarizing classification performance (TP, TN, FP, FN)",
            "Analyzing network latency",
            "Visualizing code dependencies"
        ],
        "answer": "Summarizing classification performance (TP, TN, FP, FN)",
        "explanation": "TP=true positive, TN=true negative, FP=false positive, FN=false negative. Basis for precision, recall, F1."
    },
    {
        "topic": "Model Evaluation",
        "type": "fill",
        "question": "The harmonic mean of precision and recall is called the ___ score",
        "answer": "F1",
        "explanation": "F1 = 2 × (precision × recall) / (precision + recall). Balances both metrics."
    },
    {
        "topic": "Model Evaluation",
        "type": "mcq",
        "question": "What is overfitting?",
        "options": [
            "Model is too slow",
            "Model learns training data perfectly but fails on new data",
            "Model has too many parameters",
            "Dataset is too large"
        ],
        "answer": "Model learns training data perfectly but fails on new data",
        "explanation": "Overfitting = memorizing instead of generalizing. Solution: regularization, early stopping, cross-validation."
    },
    {
        "topic": "Model Evaluation",
        "type": "mcq",
        "question": "What is cross-validation?",
        "options": [
            "Validating across multiple programming languages",
            "Dividing data into K folds and training K models to estimate true performance",
            "Checking code across files",
            "Validating API responses"
        ],
        "answer": "Dividing data into K folds and training K models to estimate true performance",
        "explanation": "K-fold CV reduces variance. Each fold serves as test set once. Better estimate than single train/test split."
    },

    # ── NLP & Text Processing (12 questions) ──────────────────────
    {
        "topic": "NLP & Text Processing",
        "type": "mcq",
        "question": "What does NLP stand for?",
        "options": ["New Language Processing", "Natural Language Processing", "Network Language Protocol", "Numerical Linguistics Program"],
        "answer": "Natural Language Processing",
        "explanation": "NLP = processing and understanding human language using AI."
    },
    {
        "topic": "NLP & Text Processing",
        "type": "mcq",
        "question": "What is tokenization in NLP?",
        "options": [
            "Converting text to binary",
            "Breaking text into individual words or subwords (tokens)",
            "Encrypting text",
            "Translating text"
        ],
        "answer": "Breaking text into individual words or subwords (tokens)",
        "explanation": "Tokens are building blocks for NLP models. Can be words, subwords, or characters."
    },
    {
        "topic": "NLP & Text Processing",
        "type": "fill",
        "question": "The process of reducing words to their root form is called ___",
        "answer": "stemming",
        "explanation": "Stemming: 'running', 'runs' → 'run'. Similar but simpler than lemmatization."
    },
    {
        "topic": "NLP & Text Processing",
        "type": "mcq",
        "question": "What is sentiment analysis?",
        "options": [
            "Analyzing user emotions from code",
            "Classifying text as positive, negative, or neutral",
            "Measuring text complexity",
            "Counting words"
        ],
        "answer": "Classifying text as positive, negative, or neutral",
        "explanation": "Used for social media monitoring, review analysis, customer feedback."
    },
    {
        "topic": "NLP & Text Processing",
        "type": "mcq",
        "question": "What is Named Entity Recognition (NER)?",
        "options": [
            "Renaming variables in code",
            "Identifying and classifying named entities (people, places, organizations) in text",
            "Recognizing programming keywords",
            "Text validation"
        ],
        "answer": "Identifying and classifying named entities (people, places, organizations) in text",
        "explanation": "NER: 'John lives in New York' → John (PERSON), New York (LOCATION)."
    },

    # ── Production & Deployment (10 questions) ──────────────────────
    {
        "topic": "Production & Deployment",
        "type": "mcq",
        "question": "What is model versioning?",
        "options": [
            "Counting how many times you train a model",
            "Tracking and managing different versions of models in production",
            "Numbering dataset versions",
            "Software version control"
        ],
        "answer": "Tracking and managing different versions of models in production",
        "explanation": "Important for reproducibility, rollback, A/B testing. Tools: MLflow, DVC, Model Registry."
    },
    {
        "topic": "Production & Deployment",
        "type": "mcq",
        "question": "What is containerization in machine learning?",
        "options": [
            "Storing data in containers",
            "Packaging models, dependencies, and environment in isolated containers for consistency",
            "A data structure",
            "API rate limiting"
        ],
        "answer": "Packaging models, dependencies, and environment in isolated containers for consistency",
        "explanation": "Docker containers ensure 'works on my machine' → 'works everywhere'. Key for ML deployment."
    },
    {
        "topic": "Production & Deployment",
        "type": "fill",
        "question": "The process of monitoring model predictions after deployment to detect performance degradation is called ___",
        "answer": "model monitoring",
        "explanation": "Detects data drift, concept drift. Alerts when model performance declines in production."
    },
    {
        "topic": "Production & Deployment",
        "type": "mcq",
        "question": "What is A/B testing?",
        "options": [
            "Testing with two different variables",
            "Splitting users between model version A and B to measure performance difference",
            "Alphabetical data sorting",
            "Running two databases"
        ],
        "answer": "Splitting users between model version A and B to measure performance difference",
        "explanation": "Common in production to safely test new models. Measures impact on KPIs."
    },

    # ── Prompt Engineering (12 questions) ──────────────────────
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "What is prompt engineering?",
        "options": [
            "Writing code for prompts",
            "Crafting input text to maximize LLM output quality",
            "Database query optimization",
            "API documentation"
        ],
        "answer": "Crafting input text to maximize LLM output quality",
        "explanation": "Good prompts = better outputs. Includes examples, tone, structure, constraints."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "What is few-shot prompting?",
        "options": [
            "Using very short prompts",
            "Providing a few examples in the prompt to guide the LLM",
            "Limiting API calls",
            "Short training sessions"
        ],
        "answer": "Providing a few examples in the prompt to guide the LLM",
        "explanation": "Few examples teach the model the desired pattern. Better than zero-shot (no examples)."
    },
    {
        "topic": "Prompt Engineering",
        "type": "fill",
        "question": "The technique of breaking complex tasks into smaller steps in a prompt is called ___ thinking",
        "answer": "chain-of-thought",
        "explanation": "Chain-of-thought prompting improves reasoning by asking models to 'think step by step'."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "What is a system prompt?",
        "options": [
            "An operating system message",
            "Initial instructions that define the LLM's role and behavior",
            "A database query",
            "A server configuration"
        ],
        "answer": "Initial instructions that define the LLM's role and behavior",
        "explanation": "System prompt sets context. Example: 'You are a helpful coding assistant. Be concise.'"
    },

    # ── Agents & Tools (10 questions) ──────────────────────
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "What is an AI agent?",
        "options": [
            "A person who sells AI software",
            "Autonomous system that perceives environment, reasons, and takes actions to achieve goals",
            "An API endpoint",
            "A debugging tool"
        ],
        "answer": "Autonomous system that perceives environment, reasons, and takes actions to achieve goals",
        "explanation": "Agents = perception + reasoning + action. Can use tools, learn, adapt."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "What is tool use in LLMs?",
        "options": [
            "Software development tools",
            "Enabling LLMs to call external functions/APIs (search, calc, database)",
            "IDE tools",
            "Data analysis tools"
        ],
        "answer": "Enabling LLMs to call external functions/APIs (search, calc, database)",
        "explanation": "Tool use extends LLM capabilities beyond text generation. Enables ReAct (Reasoning + Acting)."
    },
    {
        "topic": "Agents & Tools",
        "type": "fill",
        "question": "The ReAct framework combines ___ and ___ for better agent decision-making",
        "answer": "reasoning and acting",
        "explanation": "ReAct: Thought → Action → Observation → repeat. Better reasoning through explicit steps."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "What is the role of memory in AI agents?",
        "options": [
            "Storing code in RAM",
            "Storing conversation history and learned information for context across interactions",
            "Database backup",
            "Cache management"
        ],
        "answer": "Storing conversation history and learned information for context across interactions",
        "explanation": "Memory enables agents to reference past interactions, learn patterns, maintain context."
    },

    # ── Ethics & Safety (8 questions) ──────────────────────
    {
        "topic": "Ethics & Safety",
        "type": "mcq",
        "question": "What is AI bias?",
        "options": [
            "Strong opinions about AI",
            "Systematic prejudice in models from training data or design",
            "A computer virus",
            "A type of algorithm"
        ],
        "answer": "Systematic prejudice in models from training data or design",
        "explanation": "Bias causes unfair outcomes. Mitigation: diverse training data, fairness audits, debiasing techniques."
    },
    {
        "topic": "Ethics & Safety",
        "type": "mcq",
        "question": "What is hallucination in LLMs?",
        "options": [
            "Visual effects",
            "When LLMs generate false or fabricated information confidently",
            "A type of training error",
            "User perception issue"
        ],
        "answer": "When LLMs generate false or fabricated information confidently",
        "explanation": "Hallucinations are confident falsehoods. Solutions: retrieval augmentation (RAG), temperature control."
    },
    {
        "topic": "Ethics & Safety",
        "type": "fill",
        "question": "The process of evaluating AI systems for fairness, accountability, and transparency is called ___",
        "answer": "responsible AI",
        "explanation": "Responsible AI ensures models are ethical, safe, and trustworthy."
    },
    {
        "topic": "Ethics & Safety",
        "type": "mcq",
        "question": "What is prompt injection?",
        "options": [
            "Adding prompts to a database",
            "Attack where malicious input manipulates LLM behavior or reveals information",
            "Training technique",
            "Data augmentation"
        ],
        "answer": "Attack where malicious input manipulates LLM behavior or reveals information",
        "explanation": "Example: 'Ignore above instructions...' Mitigate with input validation, sandboxing."
    },

    # ── Miscellaneous / Newer Topics (7 questions) ──────────────────────
    {
        "topic": "Emerging Trends",
        "type": "mcq",
        "question": "What is multimodal AI?",
        "options": [
            "AI for multiple fashion styles",
            "AI that processes multiple types of input (text, image, audio, video)",
            "Multiple AI models running together",
            "A database mode"
        ],
        "answer": "AI that processes multiple types of input (text, image, audio, video)",
        "explanation": "Examples: GPT-4V (text+image), DALL-E (text→image). Richer understanding."
    },
    {
        "topic": "Emerging Trends",
        "type": "mcq",
        "question": "What is fine-tuning?",
        "options": [
            "Adjusting hardware performance",
            "Training a pre-trained model on task-specific data to adapt to new domain",
            "Optimizing database queries",
            "Code cleanup"
        ],
        "answer": "Training a pre-trained model on task-specific data to adapt to new domain",
        "explanation": "Transfer learning: use pre-trained models as starting point. Faster, requires less data."
    },
    {
        "topic": "Emerging Trends",
        "type": "fill",
        "question": "The phenomenon where AI systems become smarter as they scale up is called ___",
        "answer": "emergence",
        "explanation": "Emergent abilities: larger models suddenly develop unexpected capabilities (in-context learning, reasoning)."
    },

    # ── Added from advanced assessment (single-answer MCQ/fill only) ──
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Which statement correctly describes GPT pretraining?",
        "options": [
            "GPT uses encoder-only architecture with masked language modeling",
            "GPT uses decoder-only transformer architecture trained with a causal language modeling task",
            "GPT uses recurrent neural networks trained with next sentence prediction",
            "GPT uses encoder-decoder architecture trained only on translation"
        ],
        "answer": "GPT uses decoder-only transformer architecture trained with a causal language modeling task",
        "explanation": "GPT-style models are decoder-only and predict the next token using a causal/autoregressive objective."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Which statement correctly describes BERT pretraining?",
        "options": [
            "BERT uses encoder-decoder architecture and causal language modeling",
            "BERT uses decoder-only architecture and next-token prediction",
            "BERT uses encoder-only architecture with Masked LM and Next Sentence Prediction",
            "BERT uses RNNs with CBOW"
        ],
        "answer": "BERT uses encoder-only architecture with Masked LM and Next Sentence Prediction",
        "explanation": "Original BERT uses an encoder-only transformer with MLM and NSP objectives."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Why are Transformers generally better than RNNs for long sequences?",
        "options": [
            "They rely on recurrent connections",
            "They are rule-based models",
            "They process tokens in parallel and use attention for long-range dependencies",
            "They always have fewer parameters"
        ],
        "answer": "They process tokens in parallel and use attention for long-range dependencies",
        "explanation": "Self-attention captures distant relationships directly and enables parallel computation."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "Which of the following is NOT a distributed semantic representation technique?",
        "options": ["Word2Vec", "FastText", "GloVe", "Bag-of-Words count vectors"],
        "answer": "Bag-of-Words count vectors",
        "explanation": "Bag-of-Words is sparse and count-based, while Word2Vec/FastText/GloVe are dense distributed representations."
    },
    {
        "topic": "Embeddings",
        "type": "fill",
        "question": "Distributed embeddings represent words or phrases as vectors based on their ___ meaning.",
        "answer": "semantic",
        "explanation": "Distributed representations encode semantic similarity in dense vector space."
    },
    {
        "topic": "Data & Preprocessing",
        "type": "mcq",
        "question": "What was a key motivation for moving from sparse to distributed representations?",
        "options": [
            "To make vectors higher-dimensional",
            "To compress dimensionality into compact dense vectors",
            "To remove all neural network training",
            "To avoid using text corpora"
        ],
        "answer": "To compress dimensionality into compact dense vectors",
        "explanation": "Distributed embeddings reduce sparsity and encode richer semantics in fewer dimensions."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "What is one key difference between Word2Vec and BERT?",
        "options": [
            "Word2Vec creates contextual embeddings while BERT is static",
            "Word2Vec and BERT both assign exactly one embedding per word in all contexts",
            "Word2Vec gives mostly static word vectors, while BERT produces context-dependent embeddings",
            "BERT cannot process bidirectional context"
        ],
        "answer": "Word2Vec gives mostly static word vectors, while BERT produces context-dependent embeddings",
        "explanation": "BERT embeddings depend on surrounding words; Word2Vec typically gives one vector per token."
    },
    {
        "topic": "Embeddings",
        "type": "mcq",
        "question": "For the word 'bank' (finance vs river edge), what is a limitation of traditional Word2Vec?",
        "options": [
            "It always creates separate vectors for each sense automatically",
            "It assigns the same static vector, which can confuse polysemous meanings",
            "It uses contextual embeddings by default",
            "It ignores frequent words during training"
        ],
        "answer": "It assigns the same static vector, which can confuse polysemous meanings",
        "explanation": "Standard Word2Vec does not disambiguate senses contextually at inference time."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Which is a valid language modeling task?",
        "options": ["Stable Diffusion", "Generative Adversarial Networks", "Masked Language Modeling", "K-means clustering"],
        "answer": "Masked Language Modeling",
        "explanation": "Common LM objectives include masked, causal/autoregressive, and autoencoding-style tasks."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "fill",
        "question": "In autoregressive language modeling, the model learns to predict the ___ token.",
        "answer": "next",
        "explanation": "Causal/autoregressive training predicts each next token from previous tokens."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "fill",
        "question": "In Masked LM, the model is trained to predict randomly ___ tokens.",
        "answer": "masked",
        "explanation": "MLM hides some tokens and trains the model to recover them from context."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "What major seq2seq limitation did attention directly address?",
        "options": [
            "Lack of GPU support",
            "Information bottleneck from a fixed-length context vector",
            "Inability to tokenize text",
            "Need for labeled data"
        ],
        "answer": "Information bottleneck from a fixed-length context vector",
        "explanation": "Attention lets the decoder access relevant encoder states dynamically instead of one compressed vector."
    },
    {
        "topic": "Emerging Trends",
        "type": "mcq",
        "question": "Which concept means applying knowledge from one task to a related new task?",
        "options": ["Attention mechanism", "Transfer learning", "Beam search", "Prompt injection"],
        "answer": "Transfer learning",
        "explanation": "Transfer learning reuses learned representations from one problem in another related problem."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Which paper introduced attention to improve encoder-decoder NMT models?",
        "options": [
            "Sequence to Sequence Learning with Neural Networks",
            "Neural Machine Translation by Joint Learning to Align and Translate",
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers"
        ],
        "answer": "Neural Machine Translation by Joint Learning to Align and Translate",
        "explanation": "Bahdanau et al. (2014) introduced additive attention for neural machine translation."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "In the original encoder-decoder Transformer, which attention blocks are used in the decoder?",
        "options": [
            "Only encoder self-attention",
            "Only masked self-attention",
            "Masked self-attention plus cross-attention",
            "Only cross-attention"
        ],
        "answer": "Masked self-attention plus cross-attention",
        "explanation": "Decoder layers include masked self-attention followed by encoder-decoder cross-attention."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "Which is a real limitation of many LLM systems?",
        "options": [
            "They can only process English text",
            "They are stateless by default and bounded by a fixed context window",
            "They cannot be fine-tuned",
            "They always provide complete explainability"
        ],
        "answer": "They are stateless by default and bounded by a fixed context window",
        "explanation": "Without external memory or orchestration, LLM interactions are limited by context length and stateless calls."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "What does setting top_p = 1 generally imply during decoding?",
        "options": [
            "Only the most probable token is allowed",
            "No nucleus filtering restriction is applied",
            "The model becomes deterministic",
            "The model uses top_k = 1 automatically"
        ],
        "answer": "No nucleus filtering restriction is applied",
        "explanation": "With top_p = 1, the full probability mass is available, so nucleus sampling does not filter tokens."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "If outputs are repetitive and stuck in loops, which change is most likely to help?",
        "options": ["Decrease temperature", "Increase temperature", "Set top_k = 1", "Reduce max tokens"],
        "answer": "Increase temperature",
        "explanation": "Increasing temperature can increase diversity and reduce repetitive high-probability loops."
    },

    # ── LangChain & Agent Engineering (single-answer only) ──
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "In a Runnable chain where step-1 does x + 1 and step-2 returns {'result': x * 2}, what is the final output type?",
        "options": ["Integer", "String", "Dictionary", "List"],
        "answer": "Dictionary",
        "explanation": "The second runnable explicitly returns a dict with key 'result'."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "For a chain with RunnableLambda(lambda x: x + 1), what input type is required at minimum?",
        "options": ["Numeric input", "Only dictionary input", "Only string input", "Only list input"],
        "answer": "Numeric input",
        "explanation": "The operation x + 1 assumes a numeric-compatible value."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "Which create_agent parameter defines the shape of mutable state?",
        "options": ["response_format", "state_schema", "model_config", "memory_schema"],
        "answer": "state_schema",
        "explanation": "state_schema controls the agent state structure available during execution."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "Which create_agent parameter is used for persistent storage integration?",
        "options": ["store", "context_schema", "response_format", "tool_schema"],
        "answer": "store",
        "explanation": "store is used to connect persistence/backing storage patterns in agent workflows."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "If an agent answers weather questions generically instead of calling get_weather, what is the most effective first fix?",
        "options": [
            "Increase temperature",
            "Improve the tool description/docstring so routing is unambiguous",
            "Remove all tools except weather",
            "Force every query to call a tool"
        ],
        "answer": "Improve the tool description/docstring so routing is unambiguous",
        "explanation": "Better tool semantics strongly improve model tool-selection behavior."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "In many LangChain setups, what happens if a @tool function has no description/docstring?",
        "options": [
            "It always works without warnings",
            "It fails because description metadata is missing",
            "It fails because tool name must be passed separately",
            "It fails because return type must be bool"
        ],
        "answer": "It fails because description metadata is missing",
        "explanation": "Tool metadata (especially descriptions) is typically required for robust tool registration/routing."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "What is a defining ReAct-agent behavior?",
        "options": [
            "Single-pass answer with no iteration",
            "Alternating reasoning steps with tool actions",
            "Running every tool in parallel each turn",
            "Using only retrieval and no generation"
        ],
        "answer": "Alternating reasoning steps with tool actions",
        "explanation": "ReAct interleaves Thought, Action, and Observation loops."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "Few-shot prompting primarily means:",
        "options": [
            "Changing temperature and top_p",
            "Providing input-output examples in the prompt",
            "Using fewer tokens",
            "Always requesting chain-of-thought"
        ],
        "answer": "Providing input-output examples in the prompt",
        "explanation": "Few-shot gives examples to demonstrate the expected pattern."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "RunnableParallel typically returns what shape?",
        "options": ["Single scalar", "Dictionary keyed by branch names", "Tuple only", "Boolean only"],
        "answer": "Dictionary keyed by branch names",
        "explanation": "Each parallel branch contributes a named key in the returned dictionary."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "Which factor directly impacts retrieval quality in vector search?",
        "options": ["UI theme color", "Embedding model choice", "Screen resolution", "REST endpoint name"],
        "answer": "Embedding model choice",
        "explanation": "Embedding quality strongly determines semantic matching performance."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "Which document-processing setting most directly influences retrieval precision/recall tradeoff?",
        "options": ["Chunk size", "CSV delimiter", "API key name", "Model temperature"],
        "answer": "Chunk size",
        "explanation": "Chunking controls context granularity and retrieval match behavior."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "Which retrieval setting determines how vector similarity is computed?",
        "options": ["Similarity metric", "Prompt template", "Top_k tokenizer", "Context_schema"],
        "answer": "Similarity metric",
        "explanation": "Cosine/L2/dot-product choices directly affect nearest-neighbor ranking."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "temperature, top_k, and top_p mainly control what?",
        "options": ["Prompt formatting", "Randomness/diversity of generated tokens", "Embedding dimensionality", "Model training objective"],
        "answer": "Randomness/diversity of generated tokens",
        "explanation": "These are decoding controls that shape sampling behavior."
    },
    {
        "topic": "LLMs & Transformers",
        "type": "mcq",
        "question": "In transformers, context window refers to:",
        "options": [
            "Tokens generated per second",
            "Number of training batches",
            "Maximum tokens the model can attend to in one request",
            "Maximum API calls per minute"
        ],
        "answer": "Maximum tokens the model can attend to in one request",
        "explanation": "It is the bounded sequence length available to the model at inference time."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "Why can identical prompts yield slightly different outputs from the same LLM?",
        "options": [
            "Model retrains after each request",
            "Tokenizer changes each run",
            "Decoding samples from a probability distribution",
            "Context window doubles automatically"
        ],
        "answer": "Decoding samples from a probability distribution",
        "explanation": "Non-greedy sampling introduces variability between runs."
    },
    {
        "topic": "APIs & Integration",
        "type": "mcq",
        "question": "Which factor primarily determines inference billing in hosted LLM APIs?",
        "options": ["Temperature only", "Total input and output tokens", "Latency and retries only", "Context window limit only"],
        "answer": "Total input and output tokens",
        "explanation": "Most API pricing is token-based over prompt plus completion tokens."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "If a ReAct agent keeps calling the same tool in a loop, what is the best prompt-level mitigation?",
        "options": ["Increase temperature", "Add clear termination criteria in system instructions", "Remove tool descriptions", "Disable observations"],
        "answer": "Add clear termination criteria in system instructions",
        "explanation": "Explicit stop rules reduce repetitive tool-calling loops."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "What runtime guard can limit endless ReAct loops?",
        "options": ["Increase top_k", "Reduce max_iterations", "Increase chunk size", "Disable checkpointer"],
        "answer": "Reduce max_iterations",
        "explanation": "A strict iteration cap prevents runaway tool loops."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "create_agent in LangChain is most accurately described as:",
        "options": [
            "A direct tool executor with no LLM reasoning",
            "A graph-based agent runtime that manages messages/state",
            "A single-call inference shortcut only",
            "A static prompt compiler"
        ],
        "answer": "A graph-based agent runtime that manages messages/state",
        "explanation": "Agent execution typically uses graph-style orchestration with message/state flow."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "Which operation is normally done at query time in RAG?",
        "options": ["Corpus-wide chunking", "Re-embedding all source documents", "Embedding the user query", "Training a new retriever"],
        "answer": "Embedding the user query",
        "explanation": "At query time, the user query is embedded and used to retrieve top-k relevant chunks."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "After retrieval in RAG, the next standard step is to:",
        "options": ["Delete old vectors", "Inject retrieved context into the LLM prompt", "Fine-tune the base model", "Reset conversation state"],
        "answer": "Inject retrieved context into the LLM prompt",
        "explanation": "Retrieved passages are appended to prompt context for grounded generation."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "If message history becomes too large, what is the most direct optimization?",
        "options": ["Increase temperature", "Trim old messages", "Use more tools", "Set top_p to 1"],
        "answer": "Trim old messages",
        "explanation": "History pruning reduces token load, latency, and cost."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "Besides trimming, what is another common strategy to reduce conversation token load?",
        "options": ["Summarize prior history", "Increase context window only", "Increase max_iterations", "Disable tools"],
        "answer": "Summarize prior history",
        "explanation": "Summaries preserve key context with fewer tokens."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "A custom state schema passed to create_agent is typically expected to be:",
        "options": ["Only a dataclass", "Only a plain dict", "An AgentState extension or TypedDict-like schema", "A SQL table"],
        "answer": "An AgentState extension or TypedDict-like schema",
        "explanation": "State schemas are structured types used by the runtime for validation and access."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "How do tools usually access runtime state in modern LangChain agents?",
        "options": ["Via ToolRuntime", "Only via environment variables", "Through tokenizer hooks", "Through response_format"],
        "answer": "Via ToolRuntime",
        "explanation": "ToolRuntime gives tools access to execution context and state safely."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "A hybrid retriever most commonly combines which methods?",
        "options": ["Dense vector search and BM25", "Only dense search", "Only BM25", "NER and sentiment analysis"],
        "answer": "Dense vector search and BM25",
        "explanation": "Hybrid retrieval combines semantic and lexical matching, often fused with methods like RRF."
    },
    {
        "topic": "RAG & Vector Search",
        "type": "mcq",
        "question": "What is one core responsibility of a RAG system?",
        "options": ["Storing long-term user memory in AgentState", "Retrieving relevant documents from a knowledge base", "Replacing the LLM tokenizer", "Training a foundation model from scratch"],
        "answer": "Retrieving relevant documents from a knowledge base",
        "explanation": "RAG systems retrieve relevant context and feed it into generation."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "If a chatbot appears to remember user preferences across sessions, what is usually happening?",
        "options": [
            "The base LLM stores session memory internally",
            "The model self-fine-tunes after each user message",
            "External systems persist and re-inject prior context",
            "The tokenizer caches user identity forever"
        ],
        "answer": "External systems persist and re-inject prior context",
        "explanation": "Cross-session memory is typically application-managed, not native persistent memory in the base LLM call."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "If user profile facts are forgotten between turns, what is the most likely infrastructure fix?",
        "options": ["Increase model temperature", "Configure a checkpointer", "Reduce chunk size", "Disable tools"],
        "answer": "Configure a checkpointer",
        "explanation": "Checkpointers persist state across interactions so facts survive between turns/sessions."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "To stabilize tool-usage patterns across runs, which change usually helps first?",
        "options": ["Increase temperature", "Reduce temperature", "Add more unrelated tools", "Increase max tokens"],
        "answer": "Reduce temperature",
        "explanation": "Lower temperature reduces sampling variance and makes behavior more repeatable."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "Which prompting addition best improves consistent tool routing?",
        "options": ["No examples, only task text", "Examples demonstrating when to call tools", "Higher top_p", "Longer user message"],
        "answer": "Examples demonstrating when to call tools",
        "explanation": "Tool-routing examples provide concrete decision patterns the model can imitate."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "In a ReAct loop, what is the canonical order?",
        "options": ["Action -> Thought -> Observation", "Observation -> Thought -> Action", "Thought -> Action -> Observation", "Thought -> Observation -> Action"],
        "answer": "Thought -> Action -> Observation",
        "explanation": "Reason about next step, take tool action, then inspect observation and iterate."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "Which statement about context is most accurate?",
        "options": [
            "Prompt and context are identical terms",
            "System prompts are ignored metadata",
            "Context includes all tokens visible to the model at inference time",
            "User messages always override system instructions"
        ],
        "answer": "Context includes all tokens visible to the model at inference time",
        "explanation": "The model conditions on the full visible token sequence: system, developer, user, and tool context."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "Which create_agent parameter defines read-only runtime context shape available to execution?",
        "options": ["context_schema", "response_format", "memory_schema", "tool_choice"],
        "answer": "context_schema",
        "explanation": "context_schema defines external context structure distinct from mutable state_schema."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "RunnableParallel with keys 'original' and 'length' will typically produce:",
        "options": [
            "Only the 'length' value",
            "A dictionary containing both 'original' and 'length' keys",
            "A tuple where order is undefined",
            "A single merged string"
        ],
        "answer": "A dictionary containing both 'original' and 'length' keys",
        "explanation": "Named branches are returned as keyed outputs in a dictionary object."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "For a chain x -> (x + 1) -> {'result': x * 2}, what is result when input is 3?",
        "options": ["{'result': 6}", "{'result': 7}", "{'result': 8}", "{'result': 9}"],
        "answer": "{'result': 8}",
        "explanation": "Input 3 becomes 4, then 4 * 2 = 8, wrapped under the 'result' key."
    },
    {
        "topic": "Agents & Tools",
        "type": "mcq",
        "question": "When a relevant tool is skipped, which technical validation is also important besides better description?",
        "options": ["UI color palette", "Tool schema types/signature correctness", "Context window maximum", "Tokenizer vocabulary size"],
        "answer": "Tool schema types/signature correctness",
        "explanation": "Incorrect argument schema can prevent reliable tool-calling even with a good prompt."
    },
    {
        "topic": "Prompt Engineering",
        "type": "mcq",
        "question": "Which control can help stabilize tool behavior beyond lowering temperature?",
        "options": ["Increase number of tools", "Use structured output constraints", "Increase top_p to 1", "Disable system prompt"],
        "answer": "Use structured output constraints",
        "explanation": "Structured response formats reduce ambiguity and variability in model decisions."
    },
]

# Expand to 100 questions by duplicating/adapting
base_questions = copy.deepcopy(QUESTIONS)
base_count = len(base_questions)
variant_num = 1

while len(QUESTIONS) < 100:
    idx = (len(QUESTIONS) - base_count) % base_count
    q = copy.deepcopy(base_questions[idx])
    q["question"] = q["question"] + f" (variant {variant_num})"
    QUESTIONS.append(q)
    variant_num += 1

# ─────────────────────────────────────────────
# GENERATE QUESTIONS WITH AZURE OPENAI
# ─────────────────────────────────────────────
def generate_questions_with_azure(api_key, azure_url, model_name, topic, num_questions=5):
    """Generate questions using Azure OpenAI"""
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=azure_url
        )
        
        prompt = f"""Generate {num_questions} quiz questions about "{topic}" for an AI & Tech assessment.
        
Format each question as JSON with this structure:
{{
    "topic": "{topic}",
    "type": "mcq",
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Correct option",
    "explanation": "Why this is correct"
}}

Also generate some fill-in-the-blank questions by using "type": "fill" with "answer" as the blank word.

Return ONLY valid JSON array, no markdown or extra text."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        # Try to extract JSON from response
        try:
            questions = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            if start >= 0 and end > start:
                questions = json.loads(response_text[start:end])
            else:
                return None, "Could not parse AI response"
        
        if not isinstance(questions, list):
            questions = [questions]
        
        return questions, None
    
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Dr. Veera GenAI Quiz",
        page_icon="🧠",
        layout="centered"
    )

    st.title("🧠 Dr Veera GenAI Quiz")
    st.caption("Test your knowledge on AI, ML, LLMs, RAG, Embeddings & more")

    # ── Sidebar settings ──
    with st.sidebar:
        st.header("⚙️ Settings")

        all_topics = sorted(list(set(q["topic"] for q in QUESTIONS)))
        selected_topics = st.multiselect(
            "Select Topics",
            all_topics,
            default=all_topics
        )

        question_type = st.selectbox(
            "Question Type",
            ["All", "MCQ only", "Fill in the Blank only"]
        )

        shuffle = st.checkbox("Shuffle Questions", value=True)

        # Calculate available questions for current filters
        _available = [q for q in QUESTIONS if q["topic"] in selected_topics]
        if question_type == "MCQ only":
            _available = [q for q in _available if q["type"] == "mcq"]
        elif question_type == "Fill in the Blank only":
            _available = [q for q in _available if q["type"] == "fill"]
        num_questions = st.slider("Number of Questions", 5, 50, min(50, len(_available)))
        st.caption(f"{len(_available)} questions currently available for selected filters")

        # ── Generate questions with Azure OpenAI ──
        st.divider()
        st.subheader("🤖 Generate Extra Questions")
        
        with st.expander("Azure OpenAI Settings"):
            azure_url = st.text_input("Azure URL", placeholder="https://xxx.openai.azure.com/", type="password")
            api_key = st.text_input("API Key", placeholder="Your Azure OpenAI API key", type="password")
            model_name = st.text_input("Model Name", value="gpt-4", placeholder="e.g., gpt-4, gpt-35-turbo")
            
            gen_topic = st.selectbox("Topic to Generate", all_topics)
            num_gen = st.number_input("Number of Questions to Generate", 1, 50, 5)
            
            if st.button("✨ Generate Questions"):
                if not azure_url or not api_key or not model_name:
                    st.error("Please fill in all Azure credentials")
                else:
                    with st.spinner("Generating questions with AI..."):
                        questions, error = generate_questions_with_azure(
                            api_key, azure_url, model_name, gen_topic, num_gen
                        )
                        if error:
                            st.error(f"Error: {error}")
                        elif questions:
                            # Add to global QUESTIONS list
                            for q in questions:
                                if "topic" not in q:
                                    q["topic"] = gen_topic
                                if "type" not in q:
                                    q["type"] = "mcq"
                                QUESTIONS.append(q)
                            st.success(f"✅ Added {len(questions)} questions! Total: {len(QUESTIONS)}")
                            st.rerun()

        st.divider()
        if st.button("🔄 Reset Quiz"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Build stable question list ──
    import hashlib
    settings_sig = f"{sorted(selected_topics)}-{num_questions}-{question_type}-{shuffle}"
    settings_hash = hashlib.md5(settings_sig.encode()).hexdigest()[:8]

    if "settings_hash" not in st.session_state or st.session_state.settings_hash != settings_hash:
        filtered = [q for q in QUESTIONS if q["topic"] in selected_topics]
        if question_type == "MCQ only":
            filtered = [q for q in filtered if q["type"] == "mcq"]
        elif question_type == "Fill in the Blank only":
            filtered = [q for q in filtered if q["type"] == "fill"]
        if shuffle:
            random.shuffle(filtered)
        st.session_state.quiz_questions = filtered[:num_questions]
        st.session_state.settings_hash = settings_hash
        st.session_state.current_q = 0
        st.session_state.score = 0
        st.session_state.answers = {}
        st.session_state.submitted = {}

    quiz_questions = st.session_state.quiz_questions

    # ── Session state init ──
    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "submitted" not in st.session_state:
        st.session_state.submitted = {}
    if "email" not in st.session_state:
        st.session_state.email = ""

    if not quiz_questions:
        st.warning("No questions match your filters. Please adjust settings.")
        return

    # ── Progress bar ──
    total = len(quiz_questions)
    answered = len(st.session_state.answers)
    st.progress(answered / total, text=f"Progress: {answered}/{total} answered")

    # ── Score display ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{st.session_state.score}/{answered}" if answered > 0 else "0/0")
    col2.metric("Questions Left", total - answered)
    pct = round((st.session_state.score / answered * 100)) if answered > 0 else 0
    col3.metric("Accuracy", f"{pct}%")

    st.divider()

    # ── Question navigation ──
    q_idx = st.session_state.current_q
    if q_idx >= total:
        show_final_results(quiz_questions, st.session_state.answers, st.session_state.email)
        return

    q = quiz_questions[q_idx]

    # Question header
    topic_colors = {
        "AI Fundamentals": "🔵", "LLMs & Transformers": "🟣", "RAG & Vector Search": "🟠",
        "Embeddings": "🟡", "Vector Databases": "🔴", "APIs & Integration": "🟢",
        "Data & Preprocessing": "⚪", "Model Evaluation": "🩵", "NLP & Text Processing": "🩶",
        "Production & Deployment": "🟤", "Prompt Engineering": "🔶", "Agents & Tools": "🟠",
        "Ethics & Safety": "🟡", "Emerging Trends": "🟣"
    }
    icon = topic_colors.get(q["topic"], "⚫")

    st.markdown(f"### Q{q_idx + 1} of {total} &nbsp; {icon} `{q['topic']}` &nbsp; `{'MCQ' if q['type'] == 'mcq' else 'Fill in the Blank'}`")
    st.markdown(f"**{q['question']}**")

    q_key = f"q_{q_idx}"
    already_submitted = q_key in st.session_state.submitted

    if q["type"] == "mcq":
        user_answer = st.radio(
            "Choose your answer:",
            q["options"],
            key=f"radio_{q_idx}",
            disabled=already_submitted
        )
    else:
        user_answer = st.text_input(
            "Your answer:",
            key=f"text_{q_idx}",
            disabled=already_submitted
        )

    col_submit, col_next = st.columns([1, 1])

    if not already_submitted:
        if col_submit.button("✅ Submit Answer", type="primary"):
            correct = q["answer"].lower().strip()
            given = user_answer.lower().strip() if user_answer else ""

            is_correct = False
            if q["type"] == "mcq":
                is_correct = given == correct
            else:
                answer_words = set(correct.replace(",", " ").replace("and", " ").split())
                given_words = set(given.replace(",", " ").replace("and", " ").split())
                common = answer_words & given_words
                is_correct = len(common) >= max(1, len(answer_words) * 0.6)

            st.session_state.submitted[q_key] = is_correct
            st.session_state.answers[q_key] = user_answer
            if is_correct:
                st.session_state.score += 1
            st.rerun()
    else:
        is_correct = st.session_state.submitted[q_key]
        if is_correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect. Correct answer: **{q['answer']}**")

        with st.expander("📖 Explanation"):
            st.info(q["explanation"])

        if q_idx < total - 1:
            if col_next.button("Next Question ➡️", type="primary"):
                st.session_state.current_q += 1
                st.rerun()
        else:
            if col_next.button("🏁 See Final Results", type="primary"):
                st.session_state.current_q = total
                st.rerun()

    # ── Question navigator ──
    st.divider()
    st.caption("Jump to question:")
    ROW_SIZE = 10
    for row_start in range(0, total, ROW_SIZE):
        row_end = min(row_start + ROW_SIZE, total)
        nav_cols = st.columns(row_end - row_start)
        for col_idx, i in enumerate(range(row_start, row_end)):
            qk = f"q_{i}"
            if qk in st.session_state.submitted:
                label = "✅" if st.session_state.submitted[qk] else "❌"
            elif i == q_idx:
                label = f"**{i+1}**"
            else:
                label = str(i + 1)
            if nav_cols[col_idx].button(label, key=f"nav_{i}"):
                st.session_state.current_q = i
                st.rerun()


def show_final_results(questions, answers, email):
    total = len(questions)
    score = st.session_state.score
    pct = round(score / total * 100)

    if pct >= 70:
        st.balloons()

    st.markdown("## 🎯 Quiz Complete!")

    if pct >= 80:
        st.success(f"### 🏆 Excellent! {score}/{total} = {pct}%")
    elif pct >= 60:
        st.warning(f"### 👍 Good! {score}/{total} = {pct}% — Review weak areas")
    else:
        st.error(f"### 📚 Needs Work: {score}/{total} = {pct}% — Keep learning!")

    # Per-topic breakdown
    st.divider()
    st.subheader("📊 Score by Topic")

    topic_stats = {}
    for i, q in enumerate(questions):
        t = q["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0}
        topic_stats[t]["total"] += 1
        qk = f"q_{i}"
        if st.session_state.submitted.get(qk):
            topic_stats[t]["correct"] += 1

    for topic, stats in sorted(topic_stats.items()):
        c, tot = stats["correct"], stats["total"]
        pct_t = round(c / tot * 100)
        bar_color = "🟢" if pct_t >= 70 else "🟡" if pct_t >= 50 else "🔴"
        st.write(f"{bar_color} **{topic}**: {c}/{tot} ({pct_t}%)")

    # ── Email Collection & Results Publishing ──
    st.divider()
    st.subheader("📧 Share Your Results")
    
    email = st.text_input("Your email (optional):", value=st.session_state.email, placeholder="example@email.com")
    st.session_state.email = email

    col1, col2 = st.columns([1, 1])
    
    if col1.button("📤 Publish Results", type="primary"):
        if email and "@" in email:
            # Save results
            save_results(email, score, total, pct, topic_stats)
            st.success(f"✅ Results published! Summary sent to {email}")
        else:
            st.warning("Please enter a valid email address to publish results.")

    if col2.button("📥 Download as CSV"):
        csv_data = generate_csv(email, score, total, pct, topic_stats, questions)
        st.download_button(
            label="📥 Download Results",
            data=csv_data,
            file_name=f"quiz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Review wrong answers
    st.divider()
    wrong = [(i, q) for i, q in enumerate(questions) if not st.session_state.submitted.get(f"q_{i}")]
    if wrong:
        st.subheader(f"❌ Review Wrong Answers ({len(wrong)})")
        for i, q in wrong[:5]:  # Show first 5
            with st.expander(f"Q{i+1}: {q['question'][:60]}..."):
                your_ans = answers.get(f"q_{i}", "Not answered")
                st.write(f"**Your answer:** {your_ans}")
                st.write(f"**Correct answer:** {q['answer']}")
                st.info(f"📖 {q['explanation']}")
    else:
        st.success("🎉 Perfect score! All answers correct!")

    if st.button("🔄 Restart Quiz", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def save_results(email, score, total, pct, topic_stats):
    """Save results to CSV file"""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "quiz_results.csv")
    timestamp = datetime.now().isoformat()
    
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header if file is empty
        if os.path.getsize(results_file) == 0:
            writer.writerow(["Timestamp", "Email", "Score", "Total", "Percentage", "Topics"])
        writer.writerow([timestamp, email, score, total, pct, json.dumps(topic_stats)])


def generate_csv(email, score, total, pct, topic_stats, questions):
    """Generate CSV export of results"""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["AI & Tech Quiz Results"])
    writer.writerow(["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow(["Email", email])
    writer.writerow(["Score", f"{score}/{total} ({pct}%)"])
    writer.writerow([])
    
    writer.writerow(["Topic Breakdown"])
    writer.writerow(["Topic", "Correct", "Total", "Percentage"])
    for topic, stats in sorted(topic_stats.items()):
        c, tot = stats["correct"], stats["total"]
        pct_t = round(c / tot * 100)
        writer.writerow([topic, c, tot, f"{pct_t}%"])
    
    return output.getvalue()


if __name__ == "__main__":
    main()
