import numpy as np


def prepare_argument(argument, string='NO_OP'):
    body = argument[::2]  # Remove all entities and keep relations
    str_idx = [i for i, x in enumerate(body) if x == string]  # Find NO_OPs
    body = [element for i, element in enumerate(body) if i not in str_idx]  # Remove NO_OPs
    return body, argument[-1]


def check_rule(body, obj, obj_string, rule, only_body):
    """
    Compare the argument with a rule.
    """
    if only_body:  # Compare only the body of the rule to the argument
        retval = (body == rule[2:])
    else:
        retval = ((body == rule[2:]) and (obj == obj_string))
    return retval


def modify_rewards(rule_list, arguments, query_rel_string, obj_string, rule_base_reward, rewards, only_body):
    rule_count = 0
    rule_count_body = 0
    for k in range(len(obj_string)):
        query_rel = query_rel_string[k]
        if query_rel in rule_list:
            rel_rules = rule_list[query_rel]
            argument_temp = [arguments[i][k] for i in range(len(arguments))]
            body, obj = prepare_argument(argument_temp)
            for j in range(len(rel_rules)):
                if check_rule(body, obj, obj_string[k], rel_rules[j], only_body):
                    add_reward = rule_base_reward * float(rel_rules[j][0])
                    rewards[k] += add_reward
                    break
            for j in range(len(rel_rules)):
                if check_rule(body, obj, obj_string[k], rel_rules[j], only_body=True):
                    rule_count_body += 1
                    if check_rule(body, obj, obj_string[k], rel_rules[j], only_body=False):
                        rule_count += 1
                    break
    return rewards, rule_count, rule_count_body
