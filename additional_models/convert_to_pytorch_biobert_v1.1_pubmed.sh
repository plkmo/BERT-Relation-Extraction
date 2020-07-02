export BERT_BASE_DIR=biobert_v1.1_pubmed
transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/model.ckpt-1000000 \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/biobert_v1.1_pubmed.bin
