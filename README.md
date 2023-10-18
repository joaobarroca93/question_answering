# Extractive Open-Domain Question-Answering (Extractive ODQA)

In this project I implemented an Extractive Open-Domain Question-Answering system. Given a question, the system (1) first retrieves a top K contexts related to the questions, and, after that, it (2) applies extractive QA on each of the contexts, generating a list of possible answers to the question, ranked by a score.

In order to run the Streamlit APP, please use the `entrypoint.ipynb` notebook (and follow the instructions defined in there).

## Table of Contents

1. [Main Components](#main)
    1. a [Indexing](#indexing)
    1. b [Extractive QA](#qa)
2. [Notebooks](#notebooks)
3. [Retrieval Results](#results)
4. [Future Improvements](#future)


<a name="main"></a>
## Main Components
The system is composed of many interlinked components. Before running the Extractive QA, we first need to index some documents that will be used during retrieval.

NOTE: in the project I am currently using the documents in `data/documents.csv` which correspond to the unique documents found in the original dataset (`data/ds_nlp_challenge.csv`).

The most important components are:
- Repositories - used to load/extract relevant data
    - `BaseRepository` - base class
    - `CsvDocumentsRepository` - a repository that reads all the unique documents to use during retrieval from a CSV file
- Database Client - used to connect to a database
    - `BaseDatabaseClient` - base class
    - `BaseVectorDatabaseClient` - base class for vector search databases
    - `ChromaDatabaseClient` - a client that can be used for connecting to a ChromaDB
- Encoders - used to encode text into a vector representation
    - `BaseEncoder` - base class
    - `RandomEncoder` - generates random vectors (used mostly when testing the code)
    - `TfIdfEncoder` - encodes text using TF-IDF
    - `SentenceTransformersEncoder` - encodes text using a Sentence Transformer
- Text Processors - used to appy some processing to the text
    - `BaseTextProcessor` - base class
    - `TextProcessor` - processes the text by applying lowercasing, accents normalization, stemming and stopwords removal
- Indexers - used to index documents in a database
    - `BaseIndexer` - base class
    - `DatabaseIndexer` - indexes documents into a database
- Retrievers - used to retrieve documents from a database
    - `BaseRetriever` - base class
    - `RandomRetriever` - retrieves k documents randomly (used mostly when testing the code)
    - `VectorSearchRetriever` - retrives k documents using vector search
- Readers - used to extract answers to speficic questions from one or more contexts
    - `BaseReader` - base class
    - `Reader` - extract answers from a list of contexts given a question
- Pipelines - used to aggregate multiple components in a pipeline
    - `BasePipeline` - base class
    - `ExtractiveQAPipeline` - given a query, runs the retrieval and reading, returning a list of possible answers

In addition, we have several important data classes:
- `Document` class to represent documents
- `RetrievalResult` class to represent each result of the retrieval step
- `Answer` class to represent each answer of the reader

Finally, we also have a `RankMetrics` class that generates metrics to evaluate a specific rank. Currently it supports th following metrics: `MRR@k`, `NDCG@k`, and `mAP@k`

<a name="indexing"></a>
### Indexing
We can use the script `index_documents.py` directly to index the documents into a Persisting ChromaDB:
```
python scripts/index_documents.py
```

Code example for indexing documents into a Chroma DB:
```python
from src.repositories import CsvDocumentsRepository
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.indexer import DatabaseIndexer

documents_repository = CsvDocumentsRepository(
    path="path_to_repo/data/documents.csv",  # path to the documents CSV file
    document_id_column="document_id",  # name of the column that has the document Ids
    document_content_column="document_content",  # name of the column that contains the actual document content
)
encoder = SentenceTransformersEncoder(
    model_filepath="path_to_repo/models/encoders/sentence-transformers/all-mpnet-base-deus"
)
client = ChromaDatabaseClient(
    collection_name="documents-all-mpnet-base-deus",  # name of the collection (we will use a all-mpnet-base-deus encoder to get document vectors)
    persist=True,
    persist_path="path_to_repo/data/chroma_db"  # path where db will be persisted
)
indexer = DatabaseIndexer(client=client, encoder=encoder)

docs = documents_repository.get_all()
indexer.index(documents=docs)
```

<a name="qa"></a>
### Extractive QA
We can launch the Extractive QA App:
```
streamlit run scripts/extractive_qa_streamlit.py
```

Code example for running Extractive QA:
```python
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.retriever import VectorSearchRetriever
from src.readers import Reader
from src.pipelines import ExtractiveQAPipeline

encoder = SentenceTransformersEncoder(
    model_filepath="path_to_repo/models/encoders/sentence-transformers/all-mpnet-base-deus"
)
client = ChromaDatabaseClient(
    collection_name="documents-all-mpnet-base-deus",  # name of the collection (we will use a all-mpnet-base-deus encoder to get document vectors)
    persist=True,
    persist_path="path_to_repo/data/chroma_db"  # path where db will be persisted
)
retriever = VectorSearchRetriever(client=client, encoder=encoder)
reader = Reader(model_filepath="deepset/roberta-base-squad2")  # I am using the pre-trained "deepset/roberta-base-squad2" model

top_k = 3  # retrieve top 3 contexts
question = "you question"
contexts = retriever.retrieve(query=question, k=top_k)
answers = reader.extract(
    question=question,
    contexts=contexts,
    include_contexts=True  # if true, returns the context associated to each answer
)

# Instead of the previous code, you can also use the ExtractiveQAPipeline directly
extractive_qa = ExtractiveQAPipeline(
    retriever=retriever,
    reader=reader
)
answers = extractive_qa.run(
    question=question,
    top_k=top_k,
    include_contexts=True
)

# Weight scores by relevance to get the final scores
weighted_answers = [(answer.content, answer.score*answer.context.relevance) for answer in answers]

print("Question: ", question)
print("Answers: ", sorted(weighted_answers, key=lambda x: -x[1]))
print()
pprint([answer.as_dict() for answer in answers])
```

```
Question:  When was the Premier League created?
Answers:  [('27 May 1992', 0.2723108695059864), ('1992–', 0.2514834761534068), ('1888', 0.09275213617686084)]

[{'content': '1888',
  'context': {'document': {'content': "The world's oldest football competition "
                                      'is the FA Cup, which was founded by C. '
                                      'W. Alcock and has been contested by '
                                      'English teams since 1872. The first '
                                      'official international football match '
                                      'also took place in 1872, between '
                                      'Scotland and England in Glasgow, again '
                                      'at the instigation of C. W. Alcock. '
                                      "England is also home to the world's "
                                      'first football league, which was '
                                      'founded in Birmingham in 1888 by Aston '
                                      'Villa director William McGregor. The '
                                      'original format contained 12 clubs from '
                                      'the Midlands and Northern England.',
                           'id': 'a48acd105746ddd6aea66d350ebc4434',
                           'length': 516,
                           'metadata': None},
              'relevance': 0.6029486060142517},
  'score': 0.15383091568946838},
 {'content': '1992–',
  'context': {'document': {'content': 'The league held its first season in '
                                      '1992–93 and was originally composed of '
                                      '22 clubs. The first ever Premier League '
                                      'goal was scored by Brian Deane of '
                                      'Sheffield United in a 2–1 win against '
                                      'Manchester United. The 22 inaugural '
                                      'members of the new Premier League were '
                                      'Arsenal, Aston Villa, Blackburn Rovers, '
                                      'Chelsea, Coventry City, Crystal Palace, '
                                      'Everton, Ipswich Town, Leeds United, '
                                      'Liverpool, Manchester City, Manchester '
                                      'United, Middlesbrough, Norwich City, '
                                      'Nottingham Forest, Oldham Athletic, '
                                      'Queens Park Rangers, Sheffield United, '
                                      'Sheffield Wednesday, Southampton, '
                                      'Tottenham Hotspur, and Wimbledon. Luton '
                                      'Town, Notts County and West Ham United '
                                      'were the three teams relegated from the '
                                      'old first division at the end of the '
                                      '1991–92 season, and did not take part '
                                      'in the inaugural Premier League season.',
                           'id': '5d642f0c8120bf4ab4ca6c74f07e6035',
                           'length': 797,
                           'metadata': None},
              'relevance': 0.6001180410385132},
  'score': 0.4190566837787628},
 {'content': '27 May 1992',
  'context': {'document': {'content': 'In 1992, the First Division clubs '
                                      'resigned from the Football League en '
                                      'masse and on 27 May 1992 the FA Premier '
                                      'League was formed as a limited company '
                                      'working out of an office at the '
                                      "Football Association's then "
                                      'headquarters in Lancaster Gate. This '
                                      'meant a break-up of the 104-year-old '
                                      'Football League that had operated until '
                                      'then with four divisions; the Premier '
                                      'League would operate with a single '
                                      'division and the Football League with '
                                      'three. There was no change in '
                                      'competition format; the same number of '
                                      'teams competed in the top flight, and '
                                      'promotion and relegation between the '
                                      'Premier League and the new First '
                                      'Division remained the same as the old '
                                      'First and Second Divisions with three '
                                      'teams relegated from the league and '
                                      'three promoted.',
                           'id': 'b3d5233436126c6452e87417270548df',
                           'length': 739,
                           'metadata': None},
              'relevance': 0.584426760673523},
  'score': 0.4659452438354492}]
```

<a name="notebooks"></a>
## Notebooks
Under the `notebooks` folder we have several notebooks that were used during the development of the project.
- `eda.ipynb` - a notebook for exploratory data analysis
- `data_preprocessing.ipynb` - a notebook to preprocesse the data and generate all the relevant data files
- `metrics.ipynb` - a notebook to implement and test the ranking metrics
- `tf_idf_encoder.ipynb` - a notebook for training a tf-idf model
- `sentencer_transformers_encoder.ipynb` - a notebook for fune-tuning senntence transformer models
- `retrieval_eval.ipynb` - a notebook to evaluate all the retrievers that were developed
- `extractive_qa.ipynb` - a notebook used to run extractive QA

<a name="results"></a>
## Retrieval Results

Retriever | Encoder | mAP@6 | MRR@6 | NDCG@6
--- | --- | --- | --- | --- |
Random | - | 0.00025 | 0.00025 | 0.00038 |
Vector Search | Random | 0.00064 | 0.00064 | 0.00078 |
Vector Search | TF-IDF | 0.201 | 0.201 | 0.227 |
Vector Search | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 0.676 | 0.676 | 0.71 |
Vector Search | [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 0.723 | 0.723 | 0.757 |
Vector Search | [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1)  | 0.729 | 0.729 | 0.759 |
Vector Search | all-mpnet-base-v2-deus (fine-tuned from [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)) | **0.731** | **0.731** | **0.764** |
Vector Search | mpnet-base-deus (fine-tuned from [mpnet-base](https://huggingface.co/microsoft/mpnet-base)) | 0.564 | 0.564 | 0.602 |

<a name="future"></a>
## Future Improvements
- ONNX runtime for faster encoding
- Abstractive QA (Generator using OpenAI ChatGPT)
    - The Reader can be leveraged as a plug-in (https://arxiv.org/abs/2305.08848)
- Hybrid retrieval
    - Keyword matching on query tokens
    - Filtering by entities found in the query (NER model and documents metadata)
- Train a model to predict score of answer based on all contexts and their relevances
    - Instead of multiplying the answer score with the relevance, we can train a model to infer this (by using a QA dataset, for example)
