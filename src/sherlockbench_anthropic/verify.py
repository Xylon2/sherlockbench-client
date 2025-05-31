from anthropic.types import TextBlock, ToolUseBlock
import json
from .prompts import make_verification_message_QA
from sherlockbench_client import destructure
from pprint import pprint

def last_brace_block(s: str) -> str:
    """
    Returns the last complete brace-enclosed block from the input string,
    including any nested braces. Unmatched braces are ignored.
    If no complete block exists, returns an empty string.
    """
    stack = []
    pairs = []
    for i, c in enumerate(s):
        if c == '{':
            stack.append(i)
        elif c == '}':
            if stack:
                start = stack.pop()
                pairs.append((start, i))
    if pairs:
        start, end = pairs[-1]
        return s[start:end+1]
    return ''

def verify(config, postfn, completionfn, messages, printer, attempt_id, v_formatter):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]
 
        verification_formatted = v_formatter(verification)

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification_formatted)

        # Anthropic 'Requests which include `tool_use` or `tool_result` blocks must define tools.'
        vmessages = [messages[-1]] + [make_verification_message_QA(verification_formatted)]

        # claude sometimes gives invalid json. retry a few times
        attempts = 0

        # to prevent UnboundLocalError later
        thoughts = ""
        expected_output = ""

        while attempts < 3:
            completion = completionfn(messages=vmessages)

            response = next((item.text for item in completion.content if isinstance(item, TextBlock)), None)

            try:
                # Claude often includes loads of other text in addition to
                # the JSON
                cleaned_response = last_brace_block(response)
                
                thoughts, expected_output = destructure(json.loads(cleaned_response), "thoughts", "expected_output")
                break

            except json.JSONDecodeError as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                print(cleaned_response)

        printer.print("\n--- LLM ---")
        printer.indented_print(thoughts, "\n")
        printer.print()
        printer.indented_print("`" + str(expected_output) + "`\n")

        vstatus = postfn("attempt-verification", {"attempt-id": attempt_id,
                                                  "prediction": expected_output})["status"]

        if vstatus in ("wrong"):
            printer.print("\n### SYSTEM: WRONG")
            return False
        else:
            printer.print("\n### SYSTEM: CORRECT")

        if vstatus in ("done"):
            break

    # if we got here all the verifications passed
    return True
