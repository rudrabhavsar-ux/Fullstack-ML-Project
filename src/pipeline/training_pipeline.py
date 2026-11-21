from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    train_path, test_path = DataIngestion().intiate_data()
    train_arr, test_arr, _ = DataTransformation().transform_initiate(train_path, test_path)
    print(ModelTrainer().initiate_model_trainer(train_arr, test_arr))