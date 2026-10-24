#!/bin/sh
# Idempotent: creates buckets that don't exist and (re)applies the lifecycle rule.
# Talks plain S3 through AWS_ENDPOINT_URL, so it works against any S3 backend.
set -eu

for bucket in datasets artifacts inference mlflow; do
    if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
        echo "bucket exists: $bucket"
    else
        aws s3api create-bucket --bucket "$bucket"
        echo "bucket created: $bucket"
    fi
done

# Raw uploads are only needed until labeling has had a chance to pick them up.
aws s3api put-bucket-lifecycle-configuration --bucket inference --lifecycle-configuration "{
  \"Rules\": [{
    \"ID\": \"expire-inference-raw\",
    \"Status\": \"Enabled\",
    \"Filter\": {\"Prefix\": \"raw/\"},
    \"Expiration\": {\"Days\": ${INFERENCE_RAW_RETENTION_DAYS}}
  }]
}"
echo "lifecycle rule applied: inference/raw/ expires after ${INFERENCE_RAW_RETENTION_DAYS} days"
