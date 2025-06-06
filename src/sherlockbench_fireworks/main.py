from datetime import datetime
from functools import partial

from openai import OpenAI

from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q, print_progress_with_estimate
from sherlockbench_client import run_with_error_handling, set_current_attempt

from .investigate import investigate
from .prompts import make_initial_messages

def create_completion(client, **kwargs):
    """closure to pre-load the model"""

    return client.chat.completions.create(
        **kwargs
    )

def investigate_and_verify(postfn, completionfn, config, run_id, cursor, attempt):
    attempt_id, arg_spec, test_limit = destructure(attempt, "attempt-id", "arg-spec", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = make_initial_messages(test_limit)
    messages, tool_call_count = investigate(config, postfn, completionfn, messages,
                                            printer, attempt_id, arg_spec, test_limit)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result

def run_benchmark(config, db_conn, cursor, run_id, attempts, start_time):
    """
    Run the Fireworks benchmark with the given parameters.
    This function is called by run_with_error_handling.
    """
    client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                    api_key=config['api-keys']['fireworks'])

    postfn = lambda *args: post(config["base-url"], run_id, *args)

    def completionfn(**kwargs):
        if "temperature" in config:
            kwargs["temperature"] = config['temperature']

        if "extra_body" in config:
            kwargs["extra_body"] = config['extra_body']

        if "max_tokens" in config:
            kwargs["max_tokens"] = config['max_tokens']

        return create_completion(client, model=config['model'], **kwargs)

    completionfn = LLMRateLimiter(rate_limit_seconds=config['rate-limit'],
                                  llmfn=completionfn,
                                  backoff_exceptions=[])

    executor_p = partial(investigate_and_verify, postfn, completionfn, config, run_id, cursor)

    for i, attempt in enumerate(attempts, 1):
        print_progress_with_estimate(i, len(attempts), start_time)

        # Track the current attempt for error handling
        set_current_attempt(attempt)

        # Process the attempt
        executor_p(attempt)

        # Clear the current attempt since we've completed processing it
        set_current_attempt(None)

    # Return the values needed for run completion
    return postfn, completionfn.total_call_count, config

def two_phase():
    # Use the centralized error handling function
    run_with_error_handling("fireworks", run_benchmark)
