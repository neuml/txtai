import os
os.environ["COHERE_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

from txtai import LLM
# https://neuml.github.io/txtai/install/
# https://neuml.github.io/txtai/pipeline/text/llm/#example
llm = LLM("command-r", method="litellm")

# print(llm("Where is one place you'd go in Washington, DC?", defaultrole="user"))
print(llm([{"role": "user", "content": "Where is one place you'd go in Washington, DC?"}]))


from txtai import Embeddings
# https://neuml.github.io/txtai/embeddings/configuration/vectors/#method
# https://docs.litellm.ai/docs/providers/cohere
embeddings = Embeddings(path="cohere/embed-english-v3.0", method="litellm")

# works with a list, dataset or generator
data = [
  "US tops 5 million confirmed virus cases",
  "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
  "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
  "The National Park Service warns against sacrificing slower friends in a bear attack",
  "Maine man wins $1M from $25 lottery ticket",
  "Make huge profits without work, earn up to $100,000 a day"
]

# create an index for the list of text
embeddings.index(data)
print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)
# run an embeddings search for each query
for query in ("feel good story", "climate change", 
    "public health story", "war", "wildlife", "asia",
    "lucky", "dishonest junk"):
  # extract uid of first result
  # search result format: (uid, score)
  uid = embeddings.search(query, 1)[0][0]
  # print text
  print("%-20s %s" % (query, data[uid]))
