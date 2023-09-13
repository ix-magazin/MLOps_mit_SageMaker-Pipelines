# Listing 1: Definieren der SageMaker Pipeline
from sagemaker.workflow.pipeline import Pipeline

  pipeline_name = f"BankingRejection"
  pipeline = Pipeline(
      name=pipeline_name,
      parameters=[],
      steps=[step_getdata, step_process, step_train, step_register_model],
  )

#pipeline.upsert(role_arn=role)
#execution = pipeline.start()

# Listing 2: Erstellen des PreprocessingStep
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearn

sagemaker_session = sagemaker.workflow.pipeline_context.PipelineSession()
default_bucket = f"{project_name}"

sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.0-1",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name=f"{project_name}Preprocess",
        role=role,
        code_location=f"s3://{default_bucket}/{project_folder}/code/preprocess/{current_time}",
    )

processor_args = sklearn_processor.run(
        inputs=[ ProcessingInput(source=step_getdata.properties.ProcessingOutputConfig.Outputs["input"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination= f"s3:/{default_bucket}/{project_folder}/{ExecutionVariables.PIPELINE_EXECUTION_ID}/{Preprocess}"
            ),
        ],
        code="main.py",
        source_dir="../banking_secondary_rejection/sagemaker/step_preprocess",
    )
    step_process= ProcessingStep(name=f"{project_name}Preprocess", step_args=processor_args)

# Listing 3: Python Script main.py für das Preprocessing
    base_dir = “/opt/ml/processing”
    input_path = “/opt/ml/processing/input/input.parquet”

    logger.info(“Loading training data”)
    loans_order_items = pd.read_parquet(input_path)

    logger.info(“Run preprocessing”)
    (
        encoded_train,
        encoded_test,
        encoder,
    ) = run_preprocessing(loans_order_items, target)

    encoded_train.to_csv(f”{base_dir}/train/train.csv”, index=False)
    encoded_test.to_csv(f”{base_dir}/test/test.csv”, index=False)

    joblib.dump(encoder, “model.joblib”)
    with tarfile.open(f”{base_dir}/scaler_model/model.tar.gz”, “w:gz”) as tar_handle:
        tar_handle.add(“model.joblib”)

# Listing 4: Definieren des Estimator Objekts
xgb_train = Estimator(
        image_uri=retrieve(
        framework=“xgboost“, region=“eu-central-1“, version=“1.0-1“, py_version=“py3“, instance_type=“ml.t2.medium“
    )
uri,
        instance_type=“ml.m4.xlarge“,
        instance_count=1,
        output_path= f“s3:/{default_bucket}/{project_folder}/{ExecutionVariables.PIPELINE_EXECUTION_ID}/{StepTrain}“
        role=role,
        rules=[Rule.sagemaker(rule_configs.create_xgboost_report())],
    )
    xgb_train.set_hyperparameters(**hyperparameters)

# Listing 5: Den TrainingStep vorbereiten
step_train = TrainingStep(
        name=f"{project_name}Train",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

# Listing 6: Erstellen des ModelStep
    xgb_model = Model(
        image_uri=image_uri_xgb,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

# Listing 7: Modelle für das Preprocessing festlegen
scaler_model = SKLearnModel(
        model_data=f”s3:/{default_bucket}{project_folder}/ \
                {ExecutionVariables.PIPELINE_EXECUTION_ID}/ \
                StepPreprocess/model.tar.gz"
        role=role,
        sagemaker_session=pipeline_session,
        entry_point="../banking_secondary_rejection/sagemaker/step_preprocess/main.py",
        framework_version="1.0-1",
)
    pipeline_model = PipelineModel(
        models=[scaler_model, xgb_model],
        role=role,
        sagemaker_session=pipeline_session,
    )

    inputs = CreateModelInput(
        instance_type="ml.t2.medium",
        accelerator_type="ml.eia1.medium",
    )

    step_model_create = CreateModelStep(
        name=f"{project_name}CreateModel",
        model=pipeline_model,
        inputs=inputs,
    )

# Listing 8: Modellregistrierung
register_args = pipeline_model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=’BankingRejectionGroupName’,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )

    step_register_model = ModelStep(
        name=f"{project_name}Register",
        step_args=register_args,
    )

# Listing 9:Modellinferenz an einem bereitgestellten Endpunkt
payload = data.to_csv(index=False, header=False).encode('utf-8')

response = sagemaker_runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload
)
result = json.loads(response['Body'].read().decode())
# 0.61
