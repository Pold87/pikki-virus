#!/bin/bash

#python trap_unify.py
#now=$(date +"%T")
#echo "trap_unify.py finished at $now" >> check.txt

#python trap_drop_duplicates.py
#now=$(date +"%T")
#echo "trap_drop_duplicates.py finished at $now" >> check.txt

#python complement_data.py
#now=$(date +"%T")
#echo "complement_data.py finished at $now" >> check.txt

#python train_fill.py
#now=$(date +"%T")
#echo "train_fill.py finished at $now" >> check.txt

#python aggregate_weather.py
#now=$(date +"%T")
#echo "aggregate_weather.py finished at $now" >> check.txt

#python predict_Num_I.py
#now=$(date +"%T")
#echo "predict_Num_I.py finished at $now" >> check.txt

#python add_pred_Num_I.py
#now=$(date +"%T")
#echo "add_pred_Num_I.py finished at $now" >> check.txt

#python predict_Num_II.py
#now=$(date +"%T")
#echo "predict_Num_II.py finished at $now" >> check.txt

python add_pred_Num_II.py
now=$(date +"%T")
echo "add_pred_Num_II.py finished at $now" >> check.txt

python predict_Wnv_I.py
now=$(date +"%T")
echo "predict_Wnv_I.py finished at $now" >> check.txt

python add_pred_Wnv_I.py
now=$(date +"%T")
echo "add_pred_Wnv_I.py finished at $now" >> check.txt

python predict_Wnv_II.py
now=$(date +"%T")
echo "predict_Wnv_II.py finished at $now" >> check.txt
