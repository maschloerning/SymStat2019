import json
import pprint


with open('notl_abs_nn.json', 'r') as f:
    datastore = json.load(f)

pp = pprint.PrettyPrinter(indent=4)

pp.pprint(datastore)