import re
import json


def get_rule_paths(rules):
    """
    Preprocess the rule list so that each rule can be described by a path in the graph.
    """
    preprocessed_rules = []
    for rule in rules:
        path = []
        lb_idx = [m.start() for m in re.finditer('\(', rule)]   # left bracket indices
        rb_idx = [m.start() for m in re.finditer('\)', rule)]   # right bracket indices
        s_idx = [m.start() for m in re.finditer(' ', rule)]   # space indices
        if rule[lb_idx[0]:rb_idx[0]+1] == '(X,Y)':
            preprocessed_rule = rule[:s_idx[1] + 1]
            if len(lb_idx) == 3:   # Rule length 2
                path.extend([rule[lb_idx[1]:lb_idx[1] + 5], rule[lb_idx[2]:lb_idx[2] + 5]])
                body1 = rule_body(rule, '(X,A)', '(A,X)', path, s_idx, lb_idx, rb_idx, ', ')
                body2 = rule_body(rule, '(A,Y)', '(Y,A)', path, s_idx, lb_idx, rb_idx, '\n')
                preprocessed_rule += body1 + body2
                preprocessed_rules.append(preprocessed_rule)
            elif len(lb_idx) == 4:   # Rule length 3
                path.extend([rule[lb_idx[1]:lb_idx[1] + 5], rule[lb_idx[2]:lb_idx[2] + 5], rule[lb_idx[3]:lb_idx[3] + 5]])
                if ('(X,A)' in path) or ('(A,X)' in path):
                    body1 = rule_body(rule, '(X,A)', '(A,X)', path, s_idx, lb_idx, rb_idx, ', ')
                    body2 = rule_body(rule, '(A,B)', '(B,A)', path, s_idx, lb_idx, rb_idx, ', ')
                    body3 = rule_body(rule, '(B,Y)', '(Y,B)', path, s_idx, lb_idx, rb_idx, '\n')
                elif ('(X,B)' in path) or ('(B,X)' in path):
                    body1 = rule_body(rule, '(X,B)', '(B,X)', path, s_idx, lb_idx, rb_idx, ', ')
                    body2 = rule_body(rule, '(B,A)', '(A,B)', path, s_idx, lb_idx, rb_idx, ', ')
                    body3 = rule_body(rule, '(A,Y)', '(Y,A)', path, s_idx, lb_idx, rb_idx, '\n')
                preprocessed_rule += body1 + body2 + body3
                preprocessed_rules.append(preprocessed_rule)
    preprocessed_rules = set(preprocessed_rules)
    return preprocessed_rules


def rule_body(rule, correct_order, incorrect_order, path, s_idx, lb_idx, rb_idx, add_string):
    """
    Create the body atom that fits in the rule path.
    Inverse relations are assumed to start with '_'.
    """
    if correct_order in path:
        idx = path.index(correct_order)
        body = rule[s_idx[idx+1]+1:rb_idx[idx+1]+1] + add_string
    elif incorrect_order in path:
        idx = path.index(incorrect_order)
        if rule[s_idx[idx+1]+1] == '_':
            body = rule[s_idx[idx+1]+2:lb_idx[idx+1]] + correct_order + add_string
        else:
            body = '_' + rule[s_idx[idx+1]+1:lb_idx[idx+1]] + correct_order + add_string
    else:
        raise Exception('Rule is not valid.', path, rule)
    return body


def create_rule_list(rules):
    """
    Create a list of rules, where the output format for a rule with two body atoms is
    [confidence, head relation, body relation, body relation].
    The format for a rule with three body atoms is analogous.
    """
    rule_list = []
    for rule in rules:
        single_rule = []
        t_idx = [m.start() for m in re.finditer('\t', rule)]  # tab indices
        b_idx = [m.start() for m in re.finditer('\(', rule)]  # bracket indices
        s_idx = [m.start() for m in re.finditer(' ', rule)]  # space indices
        conf = rule[t_idx[1] + 1:t_idx[2]]
        head = rule[t_idx[-1] + 1:b_idx[0]]
        single_rule.extend([conf, head])
        for i in range(len(b_idx) - 1):
            body = rule[s_idx[i + 1] + 1:b_idx[i + 1]]
            single_rule.append(body)
        rule_list.append(single_rule)
    rule_list = sorted(rule_list, key=lambda x: x[0], reverse=True)   # Sort by decreasing confidence.
    return rule_list


def create_rule_dict(rule_list):
    rule_dict = dict()
    for rule in rule_list:
        if rule[1] in rule_dict:
            # if len(rule_dict[rule[1]]) < 10:
            rule_dict[rule[1]].append(rule)
        else:
            rule_dict[rule[1]] = [rule]
    return rule_dict


def main():
    dir = '../../../datasets/WN18RR/'
    filepath = dir + 'rules-1000'
    outfile = dir + 'rules.txt'

    with open(filepath, 'r') as f:
        rules = f.readlines()

    preprocessed_rules = get_rule_paths(rules)
    rule_list = create_rule_list(preprocessed_rules)
    rule_dict = create_rule_dict(rule_list)

    with open(outfile, 'w') as f:
        json.dump(rule_dict, f)


if __name__ == '__main__':
    main()
