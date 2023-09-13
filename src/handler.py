#!/usr/bin/env python
""" Contains the handler function that will be called by the serverless. """

import os
import inference

import runpod
from runpod.serverless.utils.rp_validator import validate

from schema import INPUT_SCHEMA

MODEL = inference.Predictor()
MODEL.setup()


def run(job):
    """
    Run inference on the model.
    """
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    validated_input = validated_input["validated_input"]

    result = MODEL.predict(settings=validated_input)

    job_output = {"result": {"prompt": validated_input["prompt"], "completion": result}}

    return job_output


runpod.serverless.start({"handler": run})
