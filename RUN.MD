!git clone https://github.com/lamdx4/iot-prj iot-prj
!pip install -q xgboost scikit-learn imbalanced-learn pandas numpy joblib psutil
!python ./iot-prj/src/dataset_full/scripts/03_train_colab_highmem.py
!python ./iot-prj/src/dataset_full/scripts/04_evaluate_model.py
!python ./iot-prj/src/dataset_full/scripts/05_visualize_results.py