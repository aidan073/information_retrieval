from ranx import Qrels, Run, evaluate
import csv

qrels = Qrels.from_file("qrel_1.tsv", kind="trec")
run = Run.from_file("result_binary_1.tsv", kind="trec")
temp = evaluate(qrels, run, ["precision@1", "precision@5", "ndcg@5", "mrr", "map"])
with open('eval_results.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in temp.items():
        writer.writerow([key, value])
# temp = evaluate(qrels, run, "precision@5")
# res = dict(run.scores)['precision@5']
# res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
# with open('p@5_results.csv', 'w', newline='') as f2:
#     writer2 = csv.writer(f2)
#     for key, value in res.items():
#         writer2.writerow([key, value])