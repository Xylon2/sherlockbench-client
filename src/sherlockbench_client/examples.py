import os
import yaml

def load_examples():
    with open("resources/investigations.yaml", 'r') as f:
        data = yaml.safe_load(f)

    return data.get('problems', {})

def get_function_name_for_attempt(postfn, attempt_id):
    """Get the function name for a given attempt_id."""
    function_name_response = postfn("developer/problem-names", {})
    function_names = function_name_response.get("problem-names", [])

    for fn_info in function_names:
        if fn_info.get("id") == attempt_id:
            function_name = fn_info.get("function_name")
            if function_name is not None:
                return function_name
    
    raise ValueError(f"No function name found for attempt_id: {attempt_id}")
