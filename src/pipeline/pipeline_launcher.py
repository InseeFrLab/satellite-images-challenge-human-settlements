import gc
import os
import pandas as pd
import mlflow
import torch
import yaml
from yaml.loader import SafeLoader

from optim.evaluation_model import metrics_quality, run_eval_data
from data.download_data import download_s3_folder, load_data
from pipeline.instatiators_for_pipeline import (
    instantiate_dataloader,
    instantiate_dataloader_eval,
    instantiate_lightning_module,
    instantiate_trainer
)


def run_pipeline(run_name):
    """
    Runs the pipeline
    """
    # Open the file and load the file
    with open("config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    download_s3_folder()
    download_s3_folder(s3_folder='challenge_mexique/retained_indices/')

    X, y = load_data(bands=config["bands"])

    dataloaders, indices_retained = instantiate_dataloader(X, y, config)
    train_dl, valid_dl, test_dl = dataloaders
    indices_train, indices_val, indices_test = indices_retained

    # Evaluation
    X_eval, y_eval = load_data(
        bands=config["bands"], filepath="../data/test_data.h5", has_labels=False
        )
    df = pd.read_csv('../data/id_map.csv')
    ids_dict = dict(zip(df['ID'], df['id']))
    eval_dl = instantiate_dataloader_eval(X_eval, y_eval, config, ids_dict)

    torch.cuda.empty_cache()
    gc.collect()

    # train_dl.dataset[0][0].shape
    light_module, LightningModule = instantiate_lightning_module(config)
    trainer = instantiate_trainer(config, light_module)

    torch.cuda.empty_cache()
    gc.collect()

    if config["mlflow"]:
        remote_server_uri = "https://projet-slums-detection-mlflow.user.lab.sspcloud.fr"
        experiment_name = "challenge_mexicain"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
        mlflow.end_run()
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.autolog()
            mlflow.log_artifact(
                "config.yml",
                artifact_path="donnees"
            )
            for key, value in config.items():
                mlflow.log_param(key, value)

            trainer.fit(light_module, train_dl, valid_dl)

            checkpoint_path = trainer.checkpoint_callback.best_model_path
            light_module_checkpoint = LightningModule.load_from_checkpoint(checkpoint_path)

            torch.cuda.empty_cache()
            gc.collect()

            model = light_module_checkpoint.model

            best_metrics, metrics, best_threshold = metrics_quality(test_dl, model)

            accuracy, precision, recall, f1, auc = metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc", auc)

            accuracy_best, precision_best, recall_best, f1_best, auc_best = best_metrics
            mlflow.log_metric("accuracy_best", accuracy_best)
            mlflow.log_metric("precision_best", precision_best)
            mlflow.log_metric("recall_best", recall_best)
            mlflow.log_metric("f1_score_best", f1_best)
            mlflow.log_metric("auc_best", auc_best)

            mlflow.log_metric("best_threshold", best_threshold)

            if config['len data limit'] in [None, "None"]:
                len_images = 1100000
            else:
                len_images = config['len data limit']

            output_path_json = f"../data/retained_indices_{int(config['train prop']*100)}_{int(config['val prop']*100)}_{int(config['test prop']*100)}_{len_images}_{int(config['prop zeros']*100)}.json"

            mlflow.log_artifact(
                output_path_json,
                artifact_path="donnees"
            )

            if config['samplesubmission']:
                eval_submission_best, eval_submission = run_eval_data(eval_dl, model, best_threshold)

                submission_df = pd.DataFrame(list(eval_submission.items()), columns=["id", "class"])

                # Sauvegarder en fichier CSV
                output_path = "../data/SampleSubmissionPredicted.csv"
                submission_df.to_csv(output_path, index=False)

                pourcentage_class_1_eval = (submission_df['class'] == 1).sum()*100/len(submission_df)

                mlflow.log_metric("pourcentage_class_1_eval", pourcentage_class_1_eval)

                mlflow.log_artifact(
                    output_path,
                    artifact_path="donnees"
                )

                if os.path.exists(output_path):
                    os.remove(output_path)

                submission_df_best = pd.DataFrame(list(eval_submission_best.items()), columns=["id", "class"])

                # Sauvegarder en fichier CSV
                output_path_best = "../data/SampleSubmissionPredicted_best.csv"
                submission_df_best.to_csv(output_path_best, index=False)

                pourcentage_class_1_eval_best = (submission_df_best['class'] == 1).sum()*100/len(submission_df_best)

                mlflow.log_metric("pourcentage_class_1_eval_best", pourcentage_class_1_eval_best)

                mlflow.log_artifact(
                    output_path_best,
                    artifact_path="donnees"
                )

                if os.path.exists(output_path_best):
                    os.remove(output_path_best)

    else:
        trainer.fit(light_module, train_dl, valid_dl)

        checkpoint_path = trainer.checkpoint_callback.best_model_path
        light_module_checkpoint = LightningModule.load_from_checkpoint(checkpoint_path)

        torch.cuda.empty_cache()
        gc.collect()

        model = light_module_checkpoint.model

        best_metrics, metrics, best_threshold = metrics_quality(test_dl, model)

        accuracy, precision, recall, f1, auc = metrics
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"AUC: {auc:.2f}")

        accuracy_best, precision_best, recall_best, f1_best, auc_best = best_metrics
        print(f"Best accuracy: {accuracy_best:.2f}")
        print(f"Best precision: {precision_best:.2f}")
        print(f"Best recall: {recall_best:.2f}")
        print(f"Best f1 Score: {f1_best:.2f}")
        print(f"Best AUC: {auc_best:.2f}")

        print(f"Best threshold: {best_threshold:.2f}")

        if config['samplesubmission']:

            eval_submission_best, eval_submission = run_eval_data(eval_dl, model, best_threshold)

            submission_df = pd.DataFrame(list(eval_submission.items()), columns=["id", "class"])

            pourcentage_class_1_eval = (submission_df['class'] == 1).sum()*100/len(submission_df)

            print(f"Il y a {round(pourcentage_class_1_eval)}% d'images prédites '1' dans le jeu de test du challenge")

            # Sauvegarder en fichier CSV
            output_path = "../data/SampleSubmissionPredicted.csv"
            submission_df.to_csv(output_path, index=False)

            submission_df_best = pd.DataFrame(list(eval_submission_best.items()), columns=["id", "class"])

            pourcentage_class_1_eval_best = (submission_df_best['class'] == 1).sum()*100/len(submission_df_best)

            print(f"Il y a {round(pourcentage_class_1_eval_best)}% d'images prédites '1' dans le jeu de test du challenge avec le meilleur seuil")

            # Sauvegarder en fichier CSV
            output_path_best = "../data/SampleSubmissionPredicted_best.csv"
            submission_df_best.to_csv(output_path_best, index=False)
