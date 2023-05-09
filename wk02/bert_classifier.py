"""Train a review classifier with BERT and Amazon SageMaker."""
import sys
import importlib
import time

import boto3
import sagemaker
import pandas as pd
import numpy as np
import botocore
import matplotlib.pyplot as plt
from sagemaker.debugger import Rule, ProfilerRule, rule_configs
from sagemaker.debugger import DebuggerHookConfig
from sagemaker.debugger import ProfilerConfig, FrameworkProfile
from sagemaker.pytorch import PyTorch as PyTorchEstimator
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONLinesSerializer
from sagemaker.deserializers import JSONLinesDeserializer
from sagemaker.pytorch.model import PyTorchModel

config = botocore.config.Config(user_agent_extra='dlai-pds/c2/w2')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker',
                  config=config)

sm_runtime = boto3.client('sagemaker-runtime',
                          config=config)

sess = sagemaker.Session(sagemaker_client=sm,
                         sagemaker_runtime_client=sm_runtime)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

# Configure dataset, hyper-parameters and evaluation metrics
processed_train_data_s3_uri = 's3://{}/data/sentiment-train/'.format(bucket)
processed_validation_data_s3_uri = 's3://{}/data/sentiment-validation/'.format(bucket)

# !aws s3 cp --recursive ./data/sentiment-train $processed_train_data_s3_uri
# !aws s3 cp --recursive ./data/sentiment-validation $processed_validation_data_s3_uri
# !aws s3 ls --recursive $processed_train_data_s3_uri
# !aws s3 ls --recursive $processed_validation_data_s3_uri

s3_input_train_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    s3_data=processed_train_data_s3_uri  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
)

s3_input_validation_data = sagemaker.inputs.TrainingInput(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    s3_data=processed_validation_data_s3_uri  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
)

data_channels = {
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    'train': s3_input_train_data,  # Replace None
    'validation': s3_input_validation_data  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
}

max_seq_length = 128  # maximum number of input tokens passed to BERT model
freeze_bert_layer = False  # specifies the depth of training within the network
epochs = 3
learning_rate = 2e-5
train_batch_size = 256
train_steps_per_epoch = 50
validation_batch_size = 256
validation_steps_per_epoch = 50
seed = 42
run_validation = True

train_instance_count = 1
train_instance_type = 'ml.c5.9xlarge'
train_volume_size = 256
input_mode = 'File'

hyperparameters = {
    'max_seq_length': max_seq_length,
    'freeze_bert_layer': freeze_bert_layer,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'train_batch_size': train_batch_size,
    'train_steps_per_epoch': train_steps_per_epoch,
    'validation_batch_size': validation_batch_size,
    'validation_steps_per_epoch': validation_steps_per_epoch,
    'seed': seed,
    'run_validation': run_validation
}

metric_definitions = [
    {'Name': 'validation:loss', 'Regex': 'val_loss: ([0-9.]+)'},
    {'Name': 'validation:accuracy', 'Regex': 'val_acc: ([0-9.]+)'},
]

debugger_hook_config = DebuggerHookConfig(
    s3_output_path='s3://{}'.format(bucket),
)

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,
    framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/", start_step=5, num_steps=10)
)

rules = [ProfilerRule.sagemaker(rule_configs.ProfilerReport())]

# Train model

sys.path.append('src/')

import train

# reload the module if it has been previously loaded
if 'train' in sys.modules:
    importlib.reload(train)

# Ignore warnings below
config = train.configure_model()

label_0 = config.id2label[0]
label_1 = config.id2label[1]
label_2 = config.id2label[2]

updated_correctly = False

if label_0 != -1 or label_1 != 0 or label_2 != 1:
    print('#######################################################################################')
    print('Please check that the function \'configure_model\' in the file src/train.py is complete.')
    print('########################################################################################')
    raise Exception('Please check that the function \'configure_model\' in the file src/train.py is complete.')
else:
    print('##################')
    print('Updated correctly!')
    print('##################')

    updated_correctly = True

if updated_correctly:
    estimator = PyTorchEstimator(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_count=train_instance_count,
        instance_type=train_instance_type,
        volume_size=train_volume_size,
        py_version='py3',  # dynamically retrieves the correct training image (Python 3)
        framework_version='1.6.0',  # dynamically retrieves the correct training image (PyTorch)
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        input_mode=input_mode,
        debugger_hook_config=debugger_hook_config,
        profiler_config=profiler_config,
        rules=rules
    )

estimator.fit(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    inputs=data_channels,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    wait=False
)

training_job_name = estimator.latest_training_job.name

print('Training Job name: {}'.format(training_job_name))

training_job_name = estimator.latest_training_job.describe()['TrainingJobName']

print('Training Job name: {}'.format(training_job_name))

print(estimator.latest_training_job.describe().keys())

training_job_status_primary = estimator.latest_training_job.describe()['TrainingJobStatus']  # Replace None
print('Training Job status: {}'.format(training_job_status_primary))

estimator.latest_training_job.wait(logs=False)

df_metrics = estimator.training_job_analytics.dataframe()
print(df_metrics)

df_metrics.query("metric_name=='validation:accuracy'").plot(x='timestamp', y='value')

profiler_report_s3_uri = "s3://{}/{}/rule-output/ProfilerReport/profiler-output".format(bucket, training_job_name)


# !aws s3 ls $profiler_report_s3_uri/
# !aws s3 cp --recursive $profiler_report_s3_uri ./profiler_report/

# Deploy the model
class SentimentPredictor(Predictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(endpoint_name,
                         sagemaker_session=sagemaker_session,
                         serializer=JSONLinesSerializer(),
                         deserializer=JSONLinesDeserializer())


timestamp = int(time.time())

pytorch_model_name = '{}-{}-{}'.format(training_job_name, 'pt', timestamp)

model = PyTorchModel(name=pytorch_model_name,
                     model_data=estimator.model_data,
                     predictor_cls=SentimentPredictor,
                     entry_point='inference.py',
                     source_dir='src',
                     framework_version='1.6.0',
                     py_version='py3',
                     role=role)

pytorch_endpoint_name = '{}-{}-{}'.format(training_job_name, 'pt', timestamp)

print(pytorch_endpoint_name)

predictor = model.deploy(initial_instance_count=1,
                         instance_type='ml.m5.large',
                         endpoint_name=pytorch_endpoint_name)

# Test model
inputs = [
    {"features": ["I love this product!"]},
    {"features": ["OK, but not great."]},
    {"features": ["This is not the right product."]},
]

predictor = SentimentPredictor(endpoint_name=pytorch_endpoint_name,
                               sagemaker_session=sess)

predicted_classes = predictor.predict(inputs)

for predicted_class in predicted_classes:
    print("Predicted class {} with probability {}".format(predicted_class['predicted_label'],
                                                          predicted_class['probability']))
