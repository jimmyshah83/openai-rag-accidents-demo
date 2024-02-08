This demo is based on Kagglle data set of [US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

### Multiple data source


```
from langchain.prompts import ChatPromptTemplate

system_message = """Use the information from the below two sources to answer any questions.

Source 1: a SQL database about employee data
<source1>
{source1}
</source1>

Source 2: a text database of random information
<source2>
{source2}
</source2>
"""

prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])

full_chain = {
    "source1": {"question": lambda x: x["question"]} | db_chain.run,
    "source2": (lambda x: x['question']) | retriever,
    "question": lambda x: x['question'],
} | prompt | OpenAI()
```