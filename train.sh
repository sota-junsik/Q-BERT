export CUDA_VISIBLE_DEVICES=$1
export SQUAD_DIR=squad_dataset/
export OUTPUT_DIR=squad_output/

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "SQUAD_DIR : $SQUAD_DIR"
echo "OUTPUT_DIR : $OUTPUT_DIR"

python run_squad.py  --bert_model bert-base-uncased  --do_train --do_lower_case  --train_file $SQUAD_DIR/train-v1.1.json  --predict_file $SQUAD_DIR/dev-v1.1.json  --train_batch_size 12  --learning_rate 3e-5  --num_train_epochs 3.0  --max_seq_length 384  --doc_stride 128  --output_dir $OUTPUT_DIR
