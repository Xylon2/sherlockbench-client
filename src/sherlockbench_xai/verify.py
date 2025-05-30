import json
from openai import LengthFinishReasonError
from pydantic import BaseModel
from .prompts import make_verification_message_QA
from sherlockbench_client import destructure, make_schema

def verify(config, postfn, completionfn, messages, printer, attempt_id, v_formatter):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        verification_formatted = v_formatter(verification)

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification_formatted)

        vmessages = messages + [make_verification_message_QA(verification_formatted)]

        try:
            completion = completionfn(messages=vmessages,
                                      response_format={"type": "json_object",
                                                       "schema": make_schema(output_type).model_json_schema()})
        except LengthFinishReasonError as e:
            print("Caught a LengthFinishReasonError!")
            print("Completion:", e.completion)

            # well it failed so we break
            break

        try:
            response = completion.choices[0]

            thoughts, expected_output = destructure(json.loads(response.message.content), "thoughts", "expected_output")

        except json.decoder.JSONDecodeError as e:
            print("Failed to decode JSON")
            print("Error:", e)

            # well it failed so we return False
            return False

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
