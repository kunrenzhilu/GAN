## need to move the checkpoint file and the trained model into the expected log folder before train
python main.py --logs_dir=logs/log_100000 --optimizer=RMSProp --learning_rate=5e-5 --optimizer_param=0.9 --model=1 --iterations=1e5 --mode=visualize

