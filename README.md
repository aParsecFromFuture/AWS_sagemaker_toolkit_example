# Deploy Classification Model pipeline with Amazon SageMaker Training Toolkit

This repository contains a simple training script to build a custom classification model that compatible with Amazon SageMaker.

### Deploy Docker Image to ECR using GitHub Actions

For the ML pipeline and SageMaker to train and provision an endpoint for inference, you need to provide a Docker image and store it in ECR. 

#### Store AWS credentials in GitHub Secrets
In your GitHub repository, go to Settings > Secrets and add the AWS access key ID and secret access key as secrets. 

#### Set up GitHub Actions workflow
Make sure that the AWS_REGION and ECR_REPOSITORY environment variables in the .github/workflow/aws.yaml file are compatible with your ECR repository. 

#### Commit and push changes
Commit the workflow file and any other changes you've made, then push them to your GitHub repository.

#### Monitor the workflow
Go to the "Actions" tab in your GitHub repository to monitor the progress of the workflow. 
If everything is set up correctly, it should build your Docker image and push it to your AWS ECR repository whenever changes are pushed to the specified branch.

#### Check the ECR repository
After the build and deployment process is successful, you can check your ECR repository via the AWS console.

### Train custom model with Amazon SageMaker

In order to train your custom model you can use the following Python script.
```python
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session()
WORK_DIRECTORY = "data"
train_input = session.upload_data(WORK_DIRECTORY, key_prefix=prefix) # put "loan-eligibility.csv" file in your S3 bucket
image = # Your Image URI in ECR repository

estimator = Estimator(
    image,
    role,
    1,
    "ml.m5.xlarge",
    output_path="s3://{}/output".format(sess.default_bucket()),
    sagemaker_session=session,
)

estimator.fit({"train": train_input})
```
