
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

DATASET=dbpedia
LABEL_NAME_FILE=label_names.txt
TRAIN_CORPUS=train.txt
TEST_CORPUS=test.txt
TEST_LABEL=test_labels.txt
MAX_LEN=200
TRAIN_BATCH=64
ACCUM_STEP=2
EVAL_BATCH=208
GPUS=1
MCP_EPOCH=3
SELF_TRAIN_EPOCH=1
CATEGORY_VOCAB_SIZE=150

python3 src/train.py --dataset_dir datasets/${DATASET}/ --label_names_file ${LABEL_NAME_FILE} \
                    --train_file ${TRAIN_CORPUS} \
                    --test_file ${TEST_CORPUS} --test_label_file ${TEST_LABEL} \
                    --max_len ${MAX_LEN} \
                    --train_batch_size ${TRAIN_BATCH} --accum_steps ${ACCUM_STEP} --eval_batch_size ${EVAL_BATCH} \
                    --gpus ${GPUS} \
                    --mcp_epochs ${MCP_EPOCH} --self_train_epochs ${SELF_TRAIN_EPOCH} \
                    --category_vocab_size ${CATEGORY_VOCAB_SIZE}\
