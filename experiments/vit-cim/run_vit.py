import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
from aixsim_models.vit.evaluator import (
    parse_args,
    setup_logging,
    load_model_and_processor,
    compute_metrics,
    save_results
)
from aixsim_models.vit.arg_manager import create_training_arguments
from aixsim_models.vit.data import load_dataset
from chop.passes.module.transforms.cim import cim_matmul_transform_pass
import yaml
from transformers import Trainer
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    """Main training and evaluation function using HuggingFace Trainer."""
    task_args, model_args, data_args, training_args, optimizer_args, eval_args, metrics_args, job_args = parse_args()
    logger = setup_logging(job_args)
    
    # Set random seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    
    # Validate arguments
    if not (task_args.do_train or task_args.do_eval or task_args.do_predict):
        raise ValueError("You must specify at least one of --do_train, --do_eval, or --do_predict")
    
    logger.info("=" * 50)
    logger.info("ViT Model Training and Evaluation - HuggingFace Trainer")
    logger.info("=" * 50)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Do train: {task_args.do_train}")
    logger.info(f"Do eval: {task_args.do_eval}")
    logger.info(f"CIM transform: {task_args.enable_cim_transform}")
    if task_args.enable_cim_transform:
        logger.info(f"CIM config: {task_args.cim_config_path}")
    logger.info("=" * 50)
    
    try:
        # Load model and processor
        model, processor, device = load_model_and_processor(task_args, model_args, data_args, training_args, logger)
        if task_args.cim_config_path is not None:
            with open(task_args.cim_config_path, 'r') as f:
                q_config = yaml.safe_load(f)

            model, _ = cim_matmul_transform_pass(model, q_config)
        
        # Load datasets with processor for preprocessing
        train_dataset, eval_dataset, num_classes, dataset_info = load_dataset(
            task_args, model_args, data_args, training_args, logger, processor
        )
        
        # Create training arguments
        hf_training_args = create_training_arguments(task_args, training_args, optimizer_args, data_args, metrics_args)
        
        # Create custom LoRA trainer
        trainer = Trainer(
            model=model,
            args=hf_training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if task_args.do_eval else None,
            compute_metrics=compute_metrics if task_args.do_eval else None,
        )
        results = {}
        
        # Training
        if task_args.do_train:
            if train_dataset is None:
                raise ValueError("Training dataset is required for training")
            
            logger.info("Starting training with HuggingFace Trainer...")
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            
            # Save the model
            trainer.save_model()
            
            # Add training metrics to results
            results.update(train_result.metrics)
            logger.info("Training completed!")
        
        # Evaluation
        if task_args.do_eval:
            if eval_dataset is None:
                raise ValueError("Evaluation dataset is required for evaluation")
            
            logger.info("Starting evaluation with HuggingFace Trainer...")
            eval_result = trainer.evaluate()
            
            # Add evaluation metrics to results
            results.update(eval_result)
            logger.info("Evaluation completed!")
        
        # Prediction
        if task_args.do_predict:
            if eval_dataset is None:
                raise ValueError("Dataset is required for prediction")
            
            logger.info("Starting prediction with HuggingFace Trainer...")
            predictions = trainer.predict(eval_dataset)
            
            # Save predictions
            output_dir = Path(training_args.output_dir)
            predictions_file = output_dir / "predictions.npy"
            np.save(predictions_file, predictions.predictions)
            logger.info(f"Predictions saved to {predictions_file}")
        
        # Add metadata
        results.update({
            'model_name_or_path': model_args.model_name_or_path,
            'dataset_name': data_args.dataset_name,
            'dataset_info': {
                'num_classes': num_classes,
                'image_size': dataset_info.image_size if hasattr(dataset_info, 'image_size') else data_args.image_size,
            },
            'cim_transform_enabled': task_args.enable_cim_transform,
            'cim_config_path': task_args.cim_config_path if task_args.enable_cim_transform else None,
        })
        
        # Save results
        if results:
            save_results(results, training_args, logger)
        
        # Print final results
        logger.info("=" * 50)
        logger.info("Final Results")
        logger.info("=" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                if 'accuracy' in key:
                    logger.info(f"{key}: {value:.4f}")  # Show as decimal, not percentage
                elif 'time' in key:
                    logger.info(f"{key}: {value:.2f}ms")
                elif 'loss' in key:
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value:.6f}")
            elif isinstance(value, (int, str, bool)):
                logger.info(f"{key}: {value}")
            elif isinstance(value, dict):
                # Skip nested dictionaries in summary
                logger.info(f"{key}: [nested data]")
        
        # Also print to stdout for immediate visibility
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS:")
        print("=" * 50)
        print(eval_result["eval_accuracy"])
        print("=" * 50)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise


if __name__ == "__main__":
    main()