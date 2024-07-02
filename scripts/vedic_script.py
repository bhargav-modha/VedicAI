import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import VLLM
import csv
import os

def load_model(model_id):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    hf_auth = ''
    model_config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    print("Tokenizer initialized")
    
    return model, tokenizer, device

def initialize_chain(model, tokenizer, device):
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_text = pipeline(
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        return_full_text=True,
        task='text-generation',
        stopping_criteria=stopping_criteria,
        temperature=0.5,
        max_new_tokens=1024,
        repetition_penalty=1.2
    )

    print("Generate text initialized")

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

def load_documents():
    loader = DirectoryLoader('./data', glob="**/*.txt")
    documents = loader.load()
    return documents

def split_and_embed(documents):
    all_splits = []
    for ch_size in [500, 1000, 1500]:
      for ch_lap in [100, 200, 300]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = ch_size, chunk_overlap = ch_lap, length_function=len)
        all_splits.extend(text_splitter.split_documents(documents))

    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    if os.path.exists("faiss_index_vedic"):
        vectorstore = FAISS.load_local("faiss_index_vedic", embeddings)
        print("Loaded from local")
    else:
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        vectorstore.save_local(folder_path="faiss_index_vedic")
        print("Vectorstore created")
    
    return vectorstore

def generate_answers_for_query(query, llm, vectorstore):
    chat_history = []
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    prompt = f'''**User Inquiry:** Enclosed within the brackets [ ] below is the user's question or topic of interest. As you craft your response, imagine you are a wise sage, drawing from a vast reservoir of knowledge, including insights from Vedic literature and other ancient wisdom. Your goal is to provide a profound yet clear answer, maintaining modern English usage. Ensure your response is free from interruptions, well-structured, and easy to understand. Avoid any spelling errors.

    [ "{query}" ]

    
    **Your Response:**'''

    result = chain({"question": prompt, "chat_history": chat_history})

    answer = result['answer'].strip()
    
    return query, answer, result['source_documents'][0]

if __name__ == "__main__":
    models_list_done = ['meta-llama/Llama-2-13b-chat-hf', 'lmsys/vicuna-13b-v1.5-16k', 'tiiuae/falcon-7b-instruct']
    documents = load_documents()
    vectorstore = split_and_embed(documents)
    
    for model_id in models_list_done:
        model, tokenizer, device = load_model(model_id)
        llm = initialize_chain(model, tokenizer, device)
        queries = ["How can we apply Agni's role, as a bridge between the divine and mortal, to our personal growth and success today? Do we need a mediator, or can we connect directly with the divine?",
                "How can we balance the pursuit of material wealth and spiritual growth, as seen in the praises of Agni and Indra in the texts?",
                "According to praise of Vayu, how can we build a harmonious relationship with nature today, learning about sustainability and respecting natural resources?",
                "How does the power of songs and praises in the texts translate to the benefits of positive affirmations and gratitude in our personal development?",
                "How does pursuing knowledge and wisdom, as encouraged in the invocation of Sarasvati, contribute to a balanced and fulfilling life, ensuring alignment with ethical values?",
                "What does '''Soma''' represent in the texts, and how can its symbolism be applied to our personal and spiritual growth today?",
                "How does Indra's relationship with Soma reflect the Vedic view on the divine, nature, and life, and how can understanding this enhance our perspective?",
                "What does the recurring theme of purifying Soma symbolize about life and personal development, and how can we relate it to modern practices of self-improvement?",
                "How can we use the strength and resilience, where Soma helps conquer challenges, to overcome our own life's obstacles and achieve abundance?",
                "How does the journey of Soma in the texts reflect the Vedic worldview on life, transformation, and divinity, and can we see it as a metaphor for the human experience?",
                "How can we apply the ancient concept of making deep, rich, and pleasing sacrifices to modern personal growth and spiritual development",
                "How can the phrase '''make our food full of sweetness for us; to all the powers of sky and earth you!''' be interpreted in terms of contributing to society and finding life's purpose?",
                "How can we develop inner strength and resilience while maintaining harmonious relationships with others, as suggested in the Slokas?",
                "How does the mention of light in the sky, earth, and atmosphere, along with the request for spreading wealth, highlight the Vedic connection between nature, the divine, and prosperity, and how can we reconnect with this today for societal and environmental betterment?",
                "How does the frequent invocation of deities for success in battle reflect the socio-political climate of the Vedic period, and what values does it reveal about the society?",
                "How can the ancient teachings on prosperity following a male child's birth and family well-being be interpreted today to support gender equality and harmonious relationships?",
                "What modern practices could be considered equivalent to the sacrifices mentioned in the texts for personal growth and community contribution?",
                "Can you explain the interconnectedness and significance of specific rituals and responses mentioned in the Yajur Veda?",
                "How can the teachings on the contributions of Agni, Indra, and Surya to individual well-being be applied today for a balanced and holistic life?",
                "How can we seek protection and guardianship for our well-being today, and what role does mindfulness or spiritual practice play?",
                "What does the Slokas' mention of unsuccessful sacrifices when the offering is too large teach us about balance in personal growth and well-being?",
                "How do the interactions between deities like Surya, Brhaspati, and Agni reflect the interconnectedness of the cosmos, and what lessons can we learn for harmony with nature?",
                "How important is clarity of intent in achieving desired outcomes in spiritual and worldly endeavors, as emphasized in the Slokas?",
                "How can the principles of offerings for victory over adversaries in the texts be applied to resolving personal or communal conflicts today?",
                "How do the rituals, sacrifices, and cosmic interactions in the Slokas assist in understanding oneself, one's place in the universe, and life's ultimate purpose?",
                "How can we stay connected to the divine or universe for guidance and prosperity while remaining humble in modern life?",
                "What role can rituals and devotion play in personal growth and finding inner peace today, and how can we create meaningful rituals?",
                "How can we balance the pursuit of material success with spiritual growth and ethical living, drawing insights from the ancient texts?",
                "How can understanding life's dualities help us navigate challenges and maintain balance, drawing lessons from the ancient texts?",
                "How can we use words positively in our lives and influence others, and what practices can help us use speech for personal growth?",
                "How does the ancient perspective on leadership and responsibility compare to contemporary views, and what qualities should a good leader possess?",
                "How can an individual cultivate qualities like strength and prosperity for personal growth in today's world, and what role does self-reflection play?",
                "How does belief in divine intervention influence our understanding of personal agency and ability to influence life circumstances?",
                "How can we interpret the pursuit of excellence and leadership in modern terms, considering ethical considerations?",
                "How does the pursuit of glory and success impact life today, and how can we balance it with humility and selflessness?",
                "How can we translate ancient invocations for divine strength into personal strength when facing challenges today?",
                "What role do mentors play in personal growth today, and how can we build relationships with them?",
                "How do we balance belief in divine intervention with taking personal responsibility for our actions and decisions?",
                "How can we redefine prosperity and wealth to include emotional and spiritual abundance, not just material wealth?",
                "How can prayer and expressing gratitude contribute to personal well-being and growth, and how can we incorporate these practices daily?",
                "How can ancient wisdom from the Vedas help us protect ourselves against modern-day adversaries?",
                "What lessons can we learn from the Vedas about the connection between nature and personal growth for our own spiritual development?",
                "How can the Vedas' insights on relationships help us navigate contemporary partnerships and build harmonious connections?",
                "How do the Vedas guide us in balancing personal desires with societal expectations, and seeking fulfillment?",
                "How can modern seekers use ancient beliefs and rituals from the Vedas to guide their paths and daily lives?",
                "How can aligning our lives with the natural flow of time, as depicted in the Slokas, aid our personal growth and development?",
                "How does nurturing our inner creative fervour contribute to personal development, and how can we balance it with time and change?",
                "How does embracing the interconnectedness of all beings influence our life perspective and role in the universe?",
                "How can we find balance and contentment amidst the dualities and contradictions of life?",
                "How can I integrate teachings on time, creation, interconnectedness, and dualities into my daily life to guide my actions and decisions?"]
                        
        results = []

        for query in queries:
            results.append(generate_answers_for_query(query, llm, vectorstore))
            
        with open(f'vedic_{model_id.split("/")[1]}_answers.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Question', 'Answer'])
            for result in results:
                query, answer, source_document = result
                csv_writer.writerow([query, answer])
                print(f"Question: {query}\nAnswer: {answer}\nSource Document: {source_document}\n")
