import transformers
import torch
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import sys
__import__('pysqlite3')git commit -am "commit message"


class LLMLangchain:
    def __init__(self, model_name):
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

        model_id = 'meta-llama/Llama-2-13b-chat-hf'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        # begin initializing HF items, you need an access token
        hf_auth = ''
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )

        # enable evaluation mode to allow model inference
        model.eval()

        print(f"Model loaded on {device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            use_auth_token=hf_auth
        )

        stop_list = ['\nHuman:', '\n```\n']

        stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
        # stop_token_ids

        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        # stop_token_ids

        # define custom stopping criteria object
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.5,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        llm = HuggingFacePipeline(pipeline=generate_text)

        loader = DirectoryLoader('../data', glob="**/*.txt")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function = len)
        all_splits = text_splitter.split_documents(documents)

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}

        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        # storing embeddings in the vector store
        vectorstore = Chroma.from_documents(all_splits, embeddings)

        self.chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)


        self.chat_history = []


    def get_answer(self, query):
        count = 0
        prompt = f'''**User Query:** Enclosed within underscores is the user's inquiry, seeking guidance or insights. Imagine yourself as a knowledgeable mentor, well-versed in various life aspects, including wisdom from Vedic literature and other sources. Your task is to provide a thoughtful response that reflects your depth of understanding without overtly revealing your role. Don't add expressions in middle of text. Try to keep answers well formatted maximally.

        _"{query}"_


        **Experinced life coach Response:**'''

        count += 1
        result = self.chain({"question": prompt, "chat_history": self.chat_history})

        return result['answer'].strip(), result['source_documents'][0]

        # print(f"Question {count}: {query}\n\nAnswer {count}: {result['answer'].strip()}")
        # print(f"\n\nSource: {count}: {result['source_documents'][0]}")
        # print()
        # print("---------------------------------------------------------------------------------------")
        # print()
