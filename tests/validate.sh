#!/usr/bin/env bash
# Lints the committed SAM/CloudFormation template. Run before tagging a release.
set -euo pipefail

echo "Validating template.yaml..."
sam validate --lint --template-file template.yaml
echo "Template lint: PASS"
