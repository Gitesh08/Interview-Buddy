# Interview Buddy - Your AI-powered Interview Preparation Partner
> This repository contains the code for Interview Buddy, a Streamlit application designed to help you ace your next interview. It leverages powerful natural language processing (NLP) and machine learning (ML) techniques to provide you with a personalized and interactive interview preparation experience.

### Features:
- Ask interview-related questions: Get insightful and relevant answers to your specific interview concerns.
- Practice conversational responses: Engage in simulated interview dialogues to improve your communication skills.
- Access interview tips and resources: Leverage the knowledge base built from interview-related documents (PDFs).

### Technology Stack:
- Streamlit: Streamlit is a Python library for rapidly building web applications. We use Streamlit to create the user interface for the chatbot.
- langchain: Langchain is a collection of libraries for building conversational AI systems. It provides tools for text processing, retrieval, and interaction with large language models.
- ctransformers: Ctransformers is a library specifically designed for working with large language models (LLMs) like the Mistral-7b model used in Interview Buddy.
- Hugging Face Transformers: This library provides access to pre-trained models for various NLP tasks, including sentence transformers used for text embedding.
- PyPDF2: This library allows us to work with PDF documents, enabling Interview Buddy to access information from interview guides or preparation materials.

### Essential Libraries:
- langchain
- torch
- accelerate (optional, for multi-GPU/TPU training)
- sentence-transformers
- streamlit
- streamlit-chat
- faiss-cpu

### Supporting Libraries:
- tiktoken
- huggingface-hub
- pypdf2
- ctransformers


### Steps:

#### 1) Clone this repository to your local machine.
```
git clone https://github.com/Gitesh08/Interview-Buddy
```
#### 2) Download mistral-7b-instruct-v0.1 in same directory.
download [mistral-7b-instruct-v0.1.Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf).

#### 3) Create a Python Environment:
We recommend creating a virtual environment to isolate the dependencies of this project from your system-wide Python installation. You can use tools like venv or conda to achieve this. Here's an example using venv:
```
python -m venv interview_buddy_env
```

#### 3) Once your environment is activated, install the required libraries using pip:
```
pip install -r requirements.txt
```

#### 4) Run the application using:
```
streamlit run main.py
```
This will launch the Interview Buddy chatbot in your web browser.

**Note:** Interview Buddy currently retrieves information from interview-related documents you place in a designated directory. You can modify the code to adjust this behavior if needed.

### Contributing:
We welcome contributions to this project! Feel free to fork the repository, make changes, and submit a pull request.
