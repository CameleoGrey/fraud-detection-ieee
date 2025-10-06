
from src.research.fdi.config import config
from src.research.fdi.FraudDetectionPipeline import FraudDetectionPipeline

if __name__ == "__main__":

    pipeline = FraudDetectionPipeline(config=config)

    oof, preds = pipeline.run_full_pipeline()

    print("done")
