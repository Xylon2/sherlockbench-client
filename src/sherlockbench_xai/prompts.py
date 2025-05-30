def make_initial_messages(test_limit):
    return [
    {"role": "system", "content":
"""You are a competent and alert chatbot.

You are provided with a mystery function which you will \"interrogate\" to try to determine what it does. Use the provided tool to do this.

Once you are confident you know what the function does, you will inform the user.

n.b. it is your job to pick inputs for the mystery function. Do not ask the user to provide you with parameters to test. Test the function pro-actively with the provided tool until you work out what the mystery function does.
"""},
    {"role": "user", "content": f"""Hi. I have a mystery function and I want to find out what it does.

I would like you to test my function using the provided tool until you think you know what it does, then tell me.

You may test this function up-to {test_limit} times."""}
]

def make_decision_messages(examples_text):
    return [
        {"role": "system", "content":
         """You are a competent and alert chatbot. You will help to investigate a mystery function."""},
        {"role": "user", "content":
         f"""I have a mystery function and I want you to figure out what it does.

Here are some examples of the function's input and output:

{examples_text}

Based on these examples, please determine what the function does and summarise."""}
    ]

def make_verification_message(f_input):
    formatted = f_input

    return {"role": "user", "content":
f"""To test your theory, please tell me what is the expected output from the function with this input:

{formatted}

You no-longer have access to the tool because I am testing if you have got it right.

Please respond in JSON with two keys: \"thoughts\" and \"expected_output\".
expected_output should contain the output you expect from the function."""}

def make_verification_message_QA(f_input):
    formatted = f_input
    
    return {"role": "user", "content":
f"""To test your theory, please tell me what is the expected output from the function with this input:

{formatted}

Please respond in JSON with two keys: \"thoughts\" and \"expected_output\".
expected_output should contain the output you expect from the function."""}
