import pyterrier as pt
import os

index = pt.IndexFactory.of(os.path.join(os.getcwd(), "index"))
retriever = pt.BatchRetrieve(index)
results = retriever.search("math is")
print(results.head())