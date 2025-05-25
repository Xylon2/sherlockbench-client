import json
from sherlockbench_client import destructure
from pydantic import BaseModel

def list_to_map(input_list):
    """openai doesn't like arrays much so just assign arbritray keys"""
    keys = [chr(97 + i) for i in range(len(input_list))]  # Generate keys: 'a', 'b', 'c', etc.
    return {key: {"type": item} for key, item in zip(keys, input_list)}

def normalize_args(input_dict):
    """Converts a dict into a list of values, sorted by the alphabetical order of the keys."""
    return [input_dict[key] for key in sorted(input_dict.keys())]

def print_tool_call(printer, args, result):
    # Clean inputs to handle surrogate characters that can't be encoded
    clean_args = []
    for arg in args:
        if isinstance(arg, str):
            # Replace surrogates with replacement character
            arg = arg.encode('utf-8', 'replace').decode('utf-8')
        clean_args.append(arg)

    # Clean result similarly if it's a string
    if isinstance(result, str):
        result = result.encode('utf-8', 'replace').decode('utf-8')

    printer.indented_print(", ".join(map(str, clean_args)), "→", result)

class ToolCallHandler:
    def __init__(self, postfn, printer, attempt_id):
        self.postfn = postfn
        self.printer = printer
        self.attempt_id = attempt_id
        self.call_history = []

    def handle_tool_call(self, call):
        arguments = json.loads(call.function.arguments)
        args_norm = normalize_args(arguments)

        fnoutput, fnerror = destructure(self.postfn("test-function", {"attempt-id": self.attempt_id,
                                                                      "args": args_norm}),
                                        "output",
                                        "error")

        print_tool_call(self.printer, args_norm, fnoutput)

        if not fnerror:
            self.call_history.append((args_norm, fnoutput))

        function_call_result_message = {
            "role": "tool",
            "content": json.dumps(fnoutput),
            "tool_call_id": call.id
        }

        return function_call_result_message

    def get_call_history(self):
        return self.call_history

    def format_call_history(self):
        lines = []
        for args, output in self.call_history:
            args_str = "(" + ", ".join(map(str, args)) + ")"
            lines.append(f"{args_str} → {output}")
        return "\n".join(lines)

class NoToolException(Exception):
    """When the LLM doesn't use it's tool when it was expected to."""
    pass

class MsgLimitException(Exception):
    """When the LLM uses too many messages."""
    pass

def investigate(config, postfn, completionfn, messages, printer, attempt_id, arg_spec, test_limit):
    mapped_args = list_to_map(arg_spec)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mystery_function",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": mapped_args,
                    "required": list(mapped_args.keys()),
                    "additionalProperties": False
                },
            },
        }
    ]

    tool_handler = ToolCallHandler(postfn, printer, attempt_id)

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
