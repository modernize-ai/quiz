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
        page_title="Public AI Quiz",
        page_icon="🧠",
        layout="centered"
    )

    st.title("🧠 Public AI & Tech Quiz")
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
