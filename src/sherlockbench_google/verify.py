import sys
from sherlockbench_client import destructure, make_schema
from .utility import save_message
from .prompts import make_verification_message_QA

def verify(config, postfn, completionfn, messages, printer, attempt_id, v_formatter):
    # for each verification
    while (v_data := postfn("next-verification", {"attempt-id": attempt_id})):
        verification = v_data["next-verification"]
        output_type = v_data["output-type"]

        verification_formatted = v_formatter(verification)

        printer.print("\n### SYSTEM: inputs:")
        printer.indented_print(verification_formatted)

        vmessages = messages + [save_message("user", make_verification_message_QA(verification_formatted))]

        completion = completionfn(contents=vmessages,
                                  schema=make_schema(output_type))

        thoughts = completion.parsed.thoughts
        expected_output = completion.parsed.expected_output

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
