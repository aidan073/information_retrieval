from ranx import Qrels, Run, evaluate
import csv

qrels = Qrels.from_file("qrel_1.tsv", kind="trec")
run = Run.from_file("result_binary_2.tsv", kind="trec")
temp = evaluate(qrels, run, ["precision@1", "precision@5", "ndcg@5", "mrr", "map"])
with open('eval_results2.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in temp.items():
        writer.writerow([key, value])