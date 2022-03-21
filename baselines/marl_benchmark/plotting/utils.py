import re


def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


def get_number_agents(cols):
    finished = False
    n_agents = 0
    while not finished:
        finished = True
        for label in cols:
            if bool(re.match("(info/learner/AGENT-" + str(n_agents) + "/learner_stats/entropy)", label)):
                finished = False
                n_agents += 1
    if n_agents > 0:
        paradigm = "decentralized"
    else:
        paradigm = "centralized"
        n_agents = 1

    return n_agents, paradigm
