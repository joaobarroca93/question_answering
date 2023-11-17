# Extractive Open-Domain Question-Answering (Extractive ODQA)

This project implements an Extractive Open-Domain Question-Answering system. Given a question, the system (1) first retrieves a top K contexts related to the questions, and, after that, it (2) applies extractive QA on each of the contexts, generating a list of possible answers to the question, ranked by a score.

You can run the Streamlit App for a quick demo:
```
streamlit run app.py
```

## Table of Contents

1. [Main Components](#main)
    1. a [Indexing](#indexing)
    1. b [Extractive QA](#qa)


<a name="main"></a>
## Main Components
The system is composed of many interlinked components. Before running the Extractive QA, we first need to index some documents that will be used during retrieval.

NOTE: Currently, we are indexing the documents that compose the [Squad V2 dataset](https://huggingface.co/datasets/squad_v2).

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
from src.entities import Document
from src.indexer import DatabaseIndexer
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder


encoder = SentenceTransformersEncoder(
    model_filepath="sentence-transformers/all-mpnet-base-v2"
)
client = ChromaDatabaseClient(
    collection_name="<name_to_the_documents_collection>",
    persist=True,
    persist_path="<path_to_persist_the_chromadb>",
)
indexer = DatabaseIndexer(client=client, encoder=encoder)

# Use the Document class to instantiate the documents to index
docs = [
    Document(
        id="1",
        content="document_content",
        length=len("document_content"),
    )
]
indexer.index(documents=docs)
```

<a name="qa"></a>
### Extractive QA
We can launch the Extractive QA App:
```
streamlit run app.py
```

Code example for running Extractive QA:
```python
from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.retriever import VectorSearchRetriever
from src.readers import Reader
from src.pipelines import ExtractiveQAPipeline

encoder = SentenceTransformersEncoder(
    model_filepath="sentence-transformers/all-mpnet-base-v2"
)
client = ChromaDatabaseClient(
    collection_name="<name_to_the_documents_collection>",
    persist=True,
    persist_path="<path_to_persist_the_chromadb>",
)
retriever = VectorSearchRetriever(client=client, encoder=encoder)
reader = Reader(model_filepath="deepset/roberta-base-squad2")

top_k = 3  # retrieve top 3 contexts
question = "your question"
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

# Weight scores by relevance of its context to get the final scores
weighted_answers = [(answer.content, answer.score*answer.context.relevance) for answer in answers]

print("Question: ", question)
print("Answers: ", sorted(weighted_answers, key=lambda x: -x[1]))
print()
pprint([answer.as_dict() for answer in answers])
```

Example for the question `When was the Premier League created?`:
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
