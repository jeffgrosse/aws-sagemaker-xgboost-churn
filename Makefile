.PHONY: data bootstrap-role train evaluate deploy predict test clean

ROLE_NAME := aws-sagemaker-xgboost-churn-training-role

data:
	./scripts/download_data.sh
	python3 scripts/prepare_data.py

bootstrap-role:
	./scripts/bootstrap_training_role.sh

train:
	$(eval ROLE_ARN := $(shell aws iam get-role --role-name $(ROLE_NAME) --query Role.Arn --output text 2>/dev/null))
	@if [ -z "$(ROLE_ARN)" ]; then \
		echo "Training role not found - run 'make bootstrap-role' first."; exit 1; \
	fi
	python3 scripts/train.py --role-arn $(ROLE_ARN)

evaluate:
	python3 scripts/evaluate.py

# Reads ModelDataUrl/ImageUri from data/last_training_run.json (written by
# `make train`) so a normal train-then-deploy flow needs no manual copying.
# Override with e.g. `make deploy PARAMS='ModelDataUrl=... ImageUri=...'` to
# deploy a specific prior run instead of the most recent one.
deploy:
	sam build
	sam deploy --guided $(if $(PARAMS),--parameter-overrides "$(PARAMS)",)

predict:
	python3 scripts/invoke_endpoint.py

test:
	pytest tests/ -v
	./tests/validate.sh

clean:
	sam delete
	rm -rf .aws-sam data/processed data/last_training_run.json
