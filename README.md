# VedicAI: Conversational Agent for Vedic Literature

VedicAI pioneers an innovative conversational agent that bridges traditional wisdom and modern computational technologies, making Vedic literature more accessible and engaging. Leveraging state-of-the-art large-scale language models (LLMs) like LLaMa 2, Vicuna, and Falcon, VedicAI excels in generating accurate, thorough, and relevant responses to queries about Vedic texts.

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Experimental Setup and Results](#experimental-setup-and-results)
7. [Future Enhancements](#future-enhancements)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction
In the evolving landscape of digital humanities and artificial intelligence (AI), VedicAI bridges traditional wisdom and modern computational technologies. VedicAI employs a novel system architecture that integrates text chunking, embeddings, and Facebook AI Similarity Search (FAISS) with state-of-the-art LLMs to make Vedic literature more accessible and engaging.

## System Architecture
VedicAI's architecture includes the following components:
- **Data Layer**: Responsible for data ingestion, preprocessing, and management.
- **Intelligence Layer**: Encompasses a model pipeline with LLMs, text segmentation and embedding, and FAISS integration for text retrieval.
- **Application Layer**: Utilizes the model's outputs for specific tasks or applications.

![System Architecture](https://github.com/bhargav-modha/VedicAI/assets/56217073/7b01bd3e-e5c1-438e-a0d0-d9ca06ee93bb)

## Features
- **High Accuracy and Relevance**: Achieves high performance in accuracy (8.5/10) and relevance (9.2/10) based on human ratings.
- **Advanced System Architecture**: Integrates text chunking, embeddings, and FAISS with top-tier LLMs.
- **HPC Deployment**: Runs on high-performance computing systems to ensure efficiency and speed.
- **Cultural Heritage Preservation**: Sets new standards in AI for the preservation and accessibility of cultural heritage.
- **Future Enhancements**: Plans for multilingual support and personalized user experiences.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/VedicAI.git
    ```
2. Navigate to the project directory:
    ```bash
    cd VedicAI
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Preprocess the dataset:
    ```bash
    python preprocess.py
    ```
2. Start the application:
    ```bash
    python app.py
    ```
3. Access the chatbot interface at `http://localhost:8000`.

## Experimental Setup and Results

### Experimental Setup
The proposed architecture was implemented on a high-performance computing (HPC) system. The specifications of the system are as follows:
- **GPU**: NVIDIA Quadro GV100 with 32GB of memory
- **CPU**: Intel(R) Xeon(R) Gold 6136 at 3.00GHz
- **Memory**: 250 GB RAM

The following Python libraries were utilized:
- **Transformers** (version 4.8.2): For loading and manipulating pre-trained models.
- **LangChain** (version 1.2.0): For text document segmentation, embedding creation, and data storage in the vector store.
- **FAISS** (version 1.7.1): For similarity search and clustering of dense vectors.

Additional libraries included accelerate, beautifulsoup4, bitsandbytes, faiss-gpu, fastapi, huggingface-hub, numpy, pandas, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11, sentence-transformers, torch, transformers, and uvicorn.

### Results and Discussion
The performance of VedicAI was evaluated using 50 queries, with a focus on accuracy, thoroughness, quality, and relevance. The results are summarized as follows:

- **Accuracy**: VedicAI achieved an average human rating of 8.5/10 in accuracy. This indicates that the generated responses correctly represented the information from the selected text segments.
- **Relevance**: VedicAI received an average human rating of 9.2/10 in relevance, demonstrating the system's ability to generate responses that were highly pertinent to the user’s original queries.
- **Thoroughness and Quality**: The responses were also evaluated for thoroughness and linguistic quality, ensuring they covered key aspects of the queries and were syntactically coherent.

### Model Performance
The evaluation included a comparison of various state-of-the-art LLMs:
- **LLaMa-2-13b-chat-hf**
- **Vicuna-13b-v1.5-16k**
- **Falcon-7b**

Among these models, the **LLaMa-2-13b-chat-hf** model proved to be the most effective, delivering the best results in terms of accuracy, response quality, and computational efficiency.

## UI Screenshots
![image](https://github.com/bhargav-modha/VedicAI/assets/56217073/dd84123e-56ee-4efd-a743-e7f9e1a4ee95)
![image](https://github.com/bhargav-modha/VedicAI/assets/56217073/34d8083a-c87b-4dc3-bf70-3f4c9a89dd61)



## Future Enhancements
- **Multilingual Support**: Extending the system to support multiple languages.
- **Personalized Experiences**: Enhancing user interactions with personalized responses based on user history and preferences.

## Contributing
We welcome contributions from the community. Please open issues or submit pull requests to help us improve VedicAI.

## License
This project is licensed under the MIT License.

Explore the confluence of technology and tradition with VedicAI and gain a broader understanding of ancient wisdom!
