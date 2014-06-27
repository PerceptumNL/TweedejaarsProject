vec='weighted_tagvectorizer'

for m in euclidean cosine correlation intersection bhattacharyya
do
  echo "--- Run ${m}"
  python documentlinker.py -vectorizer $vec  -metric $m -k_link -directory '../data/final' >> report.out
  echo "--- Run ${m} with devaluation"
  python documentlinker.py -vectorizer $vec  -metric 'correlation' -tag_devaluation -link_devaluation -k_link -directory '../data/final' >> report.out
done
