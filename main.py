from src.TextSummarizer.logging import logger

from src.TextSummarizer.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.TextSummarizer.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.TextSummarizer.pipeline.stage_3_model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.TextSummarizer.pipeline.stage_4_model_evalution import ModelEvaluationTrainingPipeline
STAGE_NAME="Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")

except Exception as e:
    logger.exception(e)
    raise e
        
STAGE_NAME="Data Transformation stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_transformation_pipeline=DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"Stage {STAGE_NAME} Completed")

except Exception as e:
    logger.exception(e)
    raise e        

STAGE_NAME="Model Trainer stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_trainer_pipeline=ModelTrainerTrainingPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f"Stage {STAGE_NAME} Completed")

except Exception as e:
    logger.exception(e)
    raise e        

STAGE_NAME="Model Evaluation stage"
try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_trainer_pipeline=ModelEvaluationTrainingPipeline()
    model_trainer_pipeline.initiate_model_evaluation()
    logger.info(f"Stage {STAGE_NAME} Completed")

except Exception as e:
    logger.exception(e)
    raise e     