# mkdir -p training/json_files
# for i in $(seq -f "%03g" 0 183)
# do
python build_train_from_ranking.py \
    --tokenizer_name facebook/bart-base \
    --rank_file training/run.msmarco-passage.bm25.train.tsv \
    --json_dir training/json_files \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel training/document/msmarco-doctrain-qrels.tsv.gz \
    --query_collection training/document/msmarco-doctrain-queries.tsv \
    --doc_collection training/document/msmarco-docs.tsv
# done