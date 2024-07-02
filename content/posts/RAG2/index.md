---
title: "Expanding Beyond Basic Retrieval-Augmented Generation (RAG): Why and How"
date: 2024-07-02T10:57:08+08:00
draft: false # Set 'false' to publish
tableOfContents: true # Enable/disable Table of Contents
description: 'In the previous post, I have demonstrated the basic usage of RAG. In this post, I will include more advanced and practical tools helpful to build a stronger RAG system'
categories:
  - Tutorials
tags:
  - RAG
---

**Expanding Beyond Basic Retrieval-Augmented Generation (RAG): Why and How**

In my [previous post on RAG](https://s1monfu.github.io/posts/rag/), I introduced the concept and discussed its foundational aspects. While the basics can accommodate most use cases, today’s post aims to delve deeper into RAG, equipping you with knowledge to harness its full potential.

## Why Basic RAG is Not Enough?
**What is a 'Basic RAG'?** Essentially, it's a RAG system that relies solely on **semantic similarity search**—comparing the semantic similarity between queries and documents to retrieve the most relevant documents.

Though often sufficient, this approach has limitations. Here are some critical drawbacks of relying purely on semantic similarity:

### Drawbacks of Semantic Similarity Search
1. **Limited by Exact Keyword Retrieval**:
   - Semantic similarity search may fail to retrieve documents containing specific keywords, as it calculates similarity across the entire document. Even if a document contains the keyword, its similarity score might not highlight it.

2. **Lacks Diversity**:
   - Basic semantic search methods are **greedy**, typically retrieving documents with the highest similarity scores, which often results in similar documents clustering at the top.

To overcome these limitations, it’s crucial to explore additional techniques to enhance the effectiveness of RAG systems.

## Approaches to Amend the Shortcomings
We can address the shortcomings of basic RAG through several advanced approaches, primarily focusing on two categories: **Hybrid Search** and **Reranker**.

### Hybrid Search
Hybrid search combines multiple retrieval methods to enhance document retrieval:

1. **Keyword Search**:
   - Builds on the **TF-IDF** method, counting keyword occurrences in documents to retrieve those with the highest counts. For more on TF-IDF, refer to chapter 6.5 of this [Stanford book](https://web.stanford.edu/~jurafsky/slp3/6.pdf).

2. **Max Marginal Relevance (MMR) Search**:
   - Increases result diversity by avoiding documents similar to those already retrieved. A hyperparameter, lambda, allows for control over result diversity. Learn more about MMR in this [paper](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf).

3. **Multiple Modality Search**:
   - Allows the creation of distinct search routes for different document sections such as titles, abstracts, and content, enhancing precision.

### Reranker
After implementing a hybrid search, a reranker is essential. It reevaluates and ranks results from various modalities, unifying relevance scores to ensure the most pertinent documents are highlighted.

## Conclusion
This post explored advanced techniques in RAG, aiming to provide a clearer understanding of how to extend beyond basic capabilities. Future posts will delve into detailed implementations of these approaches. Stay tuned for more insights!