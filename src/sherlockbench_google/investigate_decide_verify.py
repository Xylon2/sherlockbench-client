import sys
import time
from google.genai import types
from .utility import save_message
from sherlockbench_client import destructure, post, AccumulatingPrinter, LLMRateLimiter, q
from datetime import datetime
from .prompts import system_message, make_initial_message, make_decision_message
from .verify import verify

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def generate_schema(input_types):
    # Generate a dictionary with keys as sequential letters and values as types.Schema objects
    schema = {
        chr(97 + i): types.Schema(type=type_str.upper())  # chr(97) is 'a', chr(98) is 'b', etc.
        for i, type_str in enumerate(input_types)
    }
    return schema

def normalize_args(input_dict):
    """Converts a dict into a list of values, sorted by the alphabetical order of the keys."""
    return [input_dict[key] for key in sorted(input_dict.keys())]

def print_tool_call(printer, args, result):
    printer.indented_print("(" + ", ".join(map(str, args)) + ")", "→", result)

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.call_history = []

    def handle_tool_call(self, call):
        arguments = call.args
        fnname = call.name
        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                       "args": args_norm}),
                                         "output",
                                         "error")

        print_tool_call(self.printer, args_norm, fnoutput)

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_response_content = types.Content(
            role='tool', parts=[types.Part.from_function_response(
                name=fnname,
                response={'result': fnoutput},
            )]
        )

        return function_response_content

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            args_str = "(" + ", ".join(map(str, args)) + ")"
            lines.append(f"{args_str} → {output}")
        return "\n".join(lines)

def get_text_from_completion(obj_list):
    """
    Concatenates the .text property from each object in the list.
    If an object doesn't have a .text property, it is skipped.

    :param obj_list: List of objects to process
    :return: Concatenated string of all .text properties
    """
    # print(f"DEBUG: obj_list type: {type(obj_list)}")
    # print(f"DEBUG: obj_list.candidates: {obj_list.candidates}")
    # print(f"DEBUG: obj_list attributes: {dir(obj_list)}")
    # print(f"DEBUG: obj_list.__dict__: {obj_list.__dict__}")
    
    if obj_list.candidates is None:
        print("DEBUG: candidates is None")
        raise RuntimeError("API returned None candidates")
    
    # print(f"DEBUG: obj_list.candidates[0]: {obj_list.candidates[0]}")
    # print(f"DEBUG: obj_list.candidates[0].content: {obj_list.candidates[0].content}")
    if obj_list.candidates[0].content is None:
        print("DEBUG: content is None")
        raise RuntimeError("API returned None content")
    
    result = ""
    for obj in obj_list.candidates[0].content.parts:
        # Use getattr with a default value to avoid AttributeError
        text = getattr(obj, "text", None)
        if text is not None:
            result += text
    return result


def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, test_limit):
    mapped_args = generate_schema(arg_spec)
    required_args = list(mapped_args.keys())
    function = types.FunctionDeclaration(
        name='mystery_function',
        description='call this function to investigate what it does',
        parameters=types.Schema(
            type='OBJECT',
            properties=mapped_args,
            required=required_args,
        ),
    )

    tools = [types.Tool(function_declarations=[function])]

    tool_handler = ToolCallHandler(postfn, printer, attempt_id)

    # call the LLM repeatedly until it stops calling it's tool
    tool_call_counter = 0
    for _ in range(0, test_limit + 5):  # the primary limit is on tool calls. This is just a failsafe
        # sometimes gemini-2.5-pro returns None
        attempts = 0
        for _ in range(3):
            completion = completionfn(contents=messages, tools=tools)

            if completion.candidates is None:
                print("Got None response. Retrying after delay.")
                time.sleep(60)
            else:
                break

        message = get_text_from_completion(completion)
        tool_calls = completion.function_calls

        printer.print("\n--- LLM ---")
        printer.indented_print(message)

        if tool_calls:
            printer.print("\n### SYSTEM: calling tool")
            for part in completion.candidates[0].content.parts:
                messages.append(part)

                if part.function_call is not None:
                    messages.append(tool_handler.handle_tool_call(part.function_call))
                    tool_call_counter += 1

        # if it didn't call the tool we can move on to verifications
        else:
            printer.print("\n### SYSTEM: The tool was used", tool_call_counter, "times.")
            messages.append(save_message("assistant", message))

            return (tool_handler.format_call_history(), tool_call_counter)

    raise MsgLimitException("Investigation loop overrun.")

def decision(completionfn, messages, printer):
    for _ in range(3):
        completion = completionfn(contents=messages)

        if completion.candidates is None:
            print("Got None response. Retrying after delay.")
            time.sleep(60)
        else:
            break

    message = get_text_from_completion(completion)

    printer.print("\n--- LLM ---")
    printer.indented_print(message)

    return messages

def investigate_decide_verify(postfn, completionfn, config, attempt, run_id, cursor):
    attempt_id, arg_spec, test_limit = destructure(attempt, "attempt-id", "arg-spec", "test-limit")

    start_time = datetime.now()
    start_api_calls = completionfn.total_call_count

    # setup the printer
    printer = AccumulatingPrinter()

    printer.print("\n### SYSTEM: interrogating function with args", arg_spec)

    messages = [save_message("user", make_initial_message(test_limit))]
    tool_calls, tool_call_count = investigate(config, postfn, completionfn, messages,
                                              printer, attempt_id, arg_spec, test_limit)
    printer.print("\n### SYSTEM: making decision based on tool calls", arg_spec)
    printer.print(tool_calls)

    messages = [save_message("user", make_decision_message(tool_calls))]
    messages = decision(completionfn, messages, printer)

    printer.print("\n### SYSTEM: verifying function with args", arg_spec)
    verification_result = verify(config, postfn, completionfn, messages, printer, attempt_id)

    time_taken = (datetime.now() - start_time).total_seconds()
    q.add_attempt(cursor, run_id, verification_result, time_taken, tool_call_count, printer, completionfn, start_api_calls, attempt_id)

    return verification_result
