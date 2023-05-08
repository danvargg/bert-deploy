"""Feature transformation with Amazon SageMaker processing job and Feature Store."""
import time
import sys
import importlib

import boto3
import sagemaker
import botocore
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import seaborn as sns

import prepare_data

config = botocore.config.Config(user_agent_extra='dlai-pds/c2/w1')

# low-level service client of the boto3 session
sm = boto3.client(service_name='sagemaker',
                  config=config)

featurestore_runtime = boto3.client(service_name='sagemaker-featurestore-runtime',
                                    config=config)

sess = sagemaker.Session(sagemaker_client=sm,
                         sagemaker_featurestore_runtime_client=featurestore_runtime)

bucket = sess.default_bucket()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

# aws s3 cp --recursive s3://dlai-practical-data-science/labs/c2w1-487391/ ./

# Configure the SageMaker Feature Store
raw_input_data_s3_uri = 's3://dlai-practical-data-science/data/raw/'
print(raw_input_data_s3_uri)

# !aws s3 ls $raw_input_data_s3_uri


timestamp = int(time.time())

feature_group_name = 'reviews-feature-group-' + str(timestamp)
feature_store_offline_prefix = 'reviews-feature-store-' + str(timestamp)

print('Feature group name: {}'.format(feature_group_name))
print('Feature store offline prefix in S3: {}'.format(feature_store_offline_prefix))

feature_definitions = [
    # unique ID of the review
    FeatureDefinition(feature_name='review_id', feature_type=FeatureTypeEnum.STRING),
    # ingestion timestamp
    FeatureDefinition(feature_name='date', feature_type=FeatureTypeEnum.STRING),
    # sentiment: -1 (negative), 0 (neutral) or 1 (positive). It will be found the Rating values (1, 2, 3, 4, 5)
    FeatureDefinition(feature_name='sentiment', feature_type=FeatureTypeEnum.STRING),
    # label ID of the target class (sentiment)
    FeatureDefinition(feature_name='label_id', feature_type=FeatureTypeEnum.STRING),
    # reviews encoded with the BERT tokenizer
    FeatureDefinition(feature_name='input_ids', feature_type=FeatureTypeEnum.STRING),
    # original Review Text
    FeatureDefinition(feature_name='review_body', feature_type=FeatureTypeEnum.STRING),
    # train/validation/test label
    FeatureDefinition(feature_name='split_type', feature_type=FeatureTypeEnum.STRING)
]

feature_group = FeatureGroup(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    name=feature_group_name,  # Replace None
    feature_definitions=feature_definitions,  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    sagemaker_session=sess
)

print(feature_group)

# Transform the dataset
processing_instance_type = 'ml.c5.xlarge'
processing_instance_count = 1
train_split_percentage = 0.90
validation_split_percentage = 0.05
test_split_percentage = 0.05
balance_dataset = True
max_seq_length = 128

processor = SKLearnProcessor(
    framework_version='0.23-1',
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    env={'AWS_DEFAULT_REGION': region},
    max_runtime_in_seconds=7200
)

# reload the module if it has been previously loaded
if 'prepare_data' in sys.modules:
    importlib.reload(prepare_data)

input_ids = prepare_data.convert_to_bert_input_ids("this product is great!", max_seq_length)

updated_correctly = False

if len(input_ids) != max_seq_length:
    print('#######################################################################################################')
    print('Please check that the function \'convert_to_bert_input_ids\' in the file src/prepare_data.py is complete.')
    print('#######################################################################################################')
    raise Exception(
        'Please check that the function \'convert_to_bert_input_ids\' in the file src/prepare_data.py is complete.')
else:
    print('##################')
    print('Updated correctly!')
    print('##################')

    updated_correctly = True

input_ids = prepare_data.convert_to_bert_input_ids("this product is great!", max_seq_length)

print(input_ids)
print('Length of the sequence: {}'.format(len(input_ids)))

if (updated_correctly):

    processor.run(code='src/prepare_data.py',
                  inputs=[
                      ProcessingInput(source=raw_input_data_s3_uri,
                                      destination='/opt/ml/processing/input/data/',
                                      s3_data_distribution_type='ShardedByS3Key')
                  ],
                  outputs=[
                      ProcessingOutput(output_name='sentiment-train',
                                       source='/opt/ml/processing/output/sentiment/train',
                                       s3_upload_mode='EndOfJob'),
                      ProcessingOutput(output_name='sentiment-validation',
                                       source='/opt/ml/processing/output/sentiment/validation',
                                       s3_upload_mode='EndOfJob'),
                      ProcessingOutput(output_name='sentiment-test',
                                       source='/opt/ml/processing/output/sentiment/test',
                                       s3_upload_mode='EndOfJob')
                  ],
                  arguments=['--train-split-percentage', str(train_split_percentage),
                             '--validation-split-percentage', str(validation_split_percentage),
                             '--test-split-percentage', str(test_split_percentage),
                             '--balance-dataset', str(balance_dataset),
                             '--max-seq-length', str(max_seq_length),
                             '--feature-store-offline-prefix', str(feature_store_offline_prefix),
                             '--feature-group-name', str(feature_group_name)
                             ],
                  logs=True,
                  wait=False)

else:
    print('#######################################')
    print('Please update the code correctly above.')
    print('#######################################')

scikit_processing_job_name = processor.jobs[-1].describe()['ProcessingJobName']

print('Processing job name: {}'.format(scikit_processing_job_name))

print(processor.jobs[-1].describe().keys())

### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
scikit_processing_job_status = processor.jobs[-1].describe()['ProcessingJobStatus']  # Replace None
### END SOLUTION - DO NOT delete this comment for grading purposes
print('Processing job status: {}'.format(scikit_processing_job_status))

running_processor = sagemaker.processing.ProcessingJob.from_processing_name(
    processing_job_name=scikit_processing_job_name,
    sagemaker_session=sess
)

running_processor.wait(logs=False)

processing_job_description = running_processor.describe()

output_config = processing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'sentiment-train':
        processed_train_data_s3_uri = output['S3Output']['S3Uri']
    if output['OutputName'] == 'sentiment-validation':
        processed_validation_data_s3_uri = output['S3Output']['S3Uri']
    if output['OutputName'] == 'sentiment-test':
        processed_test_data_s3_uri = output['S3Output']['S3Uri']

print(processed_train_data_s3_uri)
print(processed_validation_data_s3_uri)
print(processed_test_data_s3_uri)

# !aws s3 ls $processed_train_data_s3_uri/
# !aws s3 ls $processed_validation_data_s3_uri/
# !aws s3 ls $processed_test_data_s3_uri/

# !aws s3 cp $processed_train_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-train/
# !aws s3 cp $processed_validation_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-validation/
# !aws s3 cp $processed_test_data_s3_uri/part-algo-1-womens_clothing_ecommerce_reviews.tsv ./balanced/sentiment-test/

# !head -n 5 ./balanced/sentiment-train/part-algo-1-womens_clothing_ecommerce_reviews.tsv
# !head -n 5 ./balanced/sentiment-validation/part-algo-1-womens_clothing_ecommerce_reviews.tsv
# !head -n 5 ./balanced/sentiment-test/part-algo-1-womens_clothing_ecommerce_reviews.tsv

# Query the Feature Store
feature_store_query = feature_group.athena_query()

feature_store_table = feature_store_query.table_name

query_string = """
    SELECT date,
        review_id,
        sentiment, 
        label_id,
        input_ids,
        review_body
    FROM "{}" 
    WHERE split_type='train' 
    LIMIT 5
""".format(feature_store_table)

print('Glue Catalog table name: {}'.format(feature_store_table))
print('Running query: {}'.format(query_string))

output_s3_uri = 's3://{}/query_results/{}/'.format(bucket, feature_store_offline_prefix)
print(output_s3_uri)

feature_store_query.run(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    query_string=query_string,  # Replace None
    output_location=output_s3_uri  # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
)

feature_store_query.wait()

import pandas as pd

pd.set_option("max_colwidth", 100)

df_feature_store = feature_store_query.as_dataframe()
print(df_feature_store)

# Export TSV from Feature Store
df_feature_store.to_csv('./feature_store_export.tsv',
                        sep='\t',
                        index=False,
                        header=True)

# !head -n 5 ./feature_store_export.tsv
# !aws s3 cp ./feature_store_export.tsv s3://$bucket/feature_store/feature_store_export.tsv
# !aws s3 ls --recursive s3://$bucket/feature_store/feature_store_export.tsv

# Check that the dataset in the Feature Store is balanced by sentiment
feature_store_query_2 = feature_group.athena_query()

# Replace all None
query_string_count_by_sentiment = """
SELECT sentiment, COUNT(*) AS count_reviews
FROM "{}"
GROUP BY sentiment
""".format(feature_store_table)

feature_store_query_2.run(
    query_string=query_string_count_by_sentiment,  # Replace None
    output_location=output_s3_uri  # Replace None
)

feature_store_query_2.wait()

df_count_by_sentiment = feature_store_query_2.as_dataframe()
print(df_count_by_sentiment)

sns.barplot(
    data=df_count_by_sentiment,  # Replace None
    x='sentiment',  # Replace None
    y='count_reviews',  # Replace None
    color="blue"
)
