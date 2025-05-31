import json
from datetime import datetime
from functools import partial

from pydantic import BaseModel
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from sherlockbench_client.examples import load_examples, get_function_name_for_attempt

from .investigate_verify import list_to_map, normalize_args, format_tool_call, format_inputs
from .prompts import make_initial_messages, make_decision_messages
from .verify import verify

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id, arg_spec, output_type):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.arg_spec = arg_spec
        self.output_type = output_type
        self.call_history = []

    def handle_tool_call(self, call):
        arguments = json.loads(call.function.arguments)
        args_norm = normalize_args(arguments)

        try:
            fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                          "args": args_norm}),
                                            "output",
                                            "error")

            self.printer.indented_print(format_tool_call(args_norm, self.arg_spec, self.output_type, fnoutput))

            if not fnerror:
                self.call_history.append((args_norm, fnoutput))

            function_call_result_message = {
                "role": "tool",
                "content": json.dumps(fnoutput),
                "tool_call_id": call.id
            }

        except KeyError as e:
            function_call_result_message = {
                "role": "tool",
                "content": "invalid schema when calling tool",
                "tool_call_id": call.id
            }

        return function_call_result_message

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            lines.append(format_tool_call(args, self.arg_spec, self.output_type, output))
        return "\n".join(lines)

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, output_type, test_limit):
    mapped_args = list_to_map(arg_spec)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mystery_function",
                "parameters": {
                    "type": "object",
                    "properties": mapped_args,
                    "required": list(mapped_args.keys()),
                    "additionalProperties": False
                },
            },
        }
    ]

    tool_handler = ToolCallHandler(postfn, printer, attempt_id, arg_spec, output_type)

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        completion = completionfn(messages=messages, tools=tools)

        response = completion.choices[0]
        message = response.message.content
        tool_calls = response.message.tool_calls

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")
            messages.append({"role": "assistant",
                             "content": message,
                             "tool_calls": tool_calls})

            for call in tool_calls:
                messages.append(tool_handler.handle_tool_call(call))

                tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append({"role": "assistant",
                             "content": message})

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    completion = completionfn(messages=messages)

    response = completion.choices[0]
    message = response.message.content

    printer.print("\n--- LLM ---")
    printer.indented_print(message)

    return messages

def investigate_decide_verify(postfn, completionfn, config, run_id, cursor, attempt):
    attempt_id, arg_spec, output_type, test_limit = destructure(attempt, "attempt-id", "arg-spec", "output-type", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    # Get the function name for this attempt
    function_name = get_function_name_for_attempt(postfn, attempt_id)

    # Load examples for this function
    examples_data = load_examples()
    examples_text = examples_data[function_name] 

    printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
    printer.print(examples_text)

    messages = make_decision_messages(examples_text)
    messages = decision(completionfn, messages, printer)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id, partial(format_inputs, arg_spec))

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, 0, printer, completionfn, start_api_calls, attempt_id)

    return verification_result
