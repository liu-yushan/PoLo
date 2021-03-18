import os
import sys
import json
import datetime
import logging
import codecs
import resource
import gc
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from scipy.special import logsumexp as lse
from sklearn.model_selection import ParameterGrid
from mycode.options import read_options
from mycode.model.agent import Agent
from mycode.model.environment import Env
from mycode.model.baseline import ReactiveBaseline
from mycode.model.rules import prepare_argument, check_rule, modify_rewards


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
tf.compat.v1.disable_eager_execution()


class Trainer(object):
    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        self.set_random_seeds(self.seed)
        self.agent = Agent(params)
        self.train_environment = Env(params, 'train')
        self.dev_test_environment = Env(params, 'dev')
        self.test_test_environment = Env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rule_list_dir = self.input_dir + self.rule_file
        with open(self.rule_list_dir, 'r') as file:
            self.rule_list = json.load(file)
        self.baseline = ReactiveBaseline(self.Lambda)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.best_metric = -1
        self.early_stopping = False
        self.current_patience = self.patience

    def set_random_seeds(self, seed):
        if seed is not None:
            tf.compat.v1.random.set_random_seed(seed)
            np.random.seed(seed)

    def calc_reinforce_loss(self):
        loss = tf.stack(self.per_example_loss, axis=1)
        self.tf_baseline = self.baseline.get_baseline_value()

        final_rewards = self.cum_discounted_rewards - self.tf_baseline
        rewards_mean, rewards_var = tf.nn.moments(x=final_rewards, axes=[0, 1])
        rewards_std = tf.sqrt(rewards_var) + 1e-6   # Constant added for numerical stability
        final_rewards = tf.compat.v1.div(final_rewards - rewards_mean, rewards_std)

        loss = tf.multiply(loss, final_rewards)
        total_loss = tf.reduce_mean(input_tensor=loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)
        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)
        entropy_policy = - tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.multiply(tf.exp(all_logits), all_logits), axis=1))
        return entropy_policy

    def initialize(self, restore=None, sess=None):
        logger.info('Creating TF graph...')
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.query_relations = tf.compat.v1.placeholder(tf.int32, [None], name='query_relations')
        self.range_arr = tf.compat.v1.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.compat.v1.train.exponential_decay(self.beta, self.global_step, 200, 0.90,
                                                                  staircase=False)
        self.entity_sequence = []
        self.cum_discounted_rewards = tf.compat.v1.placeholder(tf.float32, [None, self.path_length],
                                                               name='cumulative_discounted_rewards')

        for t in range(self.path_length):
            next_possible_relations = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                               name='next_relations_{}'.format(t))
            next_possible_entities = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                              name='next_entities_{}'.format(t))
            start_entities = tf.compat.v1.placeholder(tf.int32, [None, ])
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)
        self.per_example_loss, self.per_example_logits, self.actions_idx = self.agent(
            self.candidate_relation_sequence, self.candidate_entity_sequence, self.entity_sequence,
            self.query_relations, self.range_arr, self.path_length)

        self.loss_op = self.calc_reinforce_loss()
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_states = tf.compat.v1.placeholder(tf.float32, self.agent.get_mem_shape(), name='memory_of_agent')
        self.prev_relations = tf.compat.v1.placeholder(tf.int32, [None, ], name='previous_relations')
        self.query_embeddings = tf.compat.v1.nn.embedding_lookup(params=self.agent.relation_lookup_table, ids=self.query_relations)
        layer_states = tf.unstack(self.prev_states, self.LSTM_layers)
        formatted_states = [tf.unstack(s, 2) for s in layer_states]
        self.next_relations = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.current_entities = tf.compat.v1.placeholder(tf.int32, shape=[None, ])

        with tf.compat.v1.variable_scope('policy_steps_unroll') as scope:
            scope.reuse_variables()
            self.test_loss, self.test_logits, test_state, self.test_actions_idx, self.chosen_relations = \
                self.agent.step(self.next_relations, self.next_entities, self.current_entities, formatted_states,
                                self.prev_relations, self.query_embeddings, self.range_arr)
            self.test_state = tf.stack(test_state)

        logger.info('TF graph creation done.')
        self.model_saver = tf.compat.v1.train.Saver(max_to_keep=2)

        if not restore:
            return tf.compat.v1.global_variables_initializer()
        else:
            return self.model_saver.restore(sess, restore)

    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_relation != '':
            logger.info('Using pretrained relation embeddings.')
            pretrained_relations = np.load(self.pretrained_embeddings_relation)
            with open(self.pretrained_relation_to_id, 'r') as f:
                relation_to_id = json.load(f)
            rel_embeddings = sess.run(self.agent.relation_lookup_table)
            for relation, idx in relation_to_id.items():
                rel_embeddings[self.relation_vocab[relation]] = pretrained_relations[idx]
            sess.run(self.agent.relation_embedding_init,
                     feed_dict={self.agent.action_embedding_placeholder: rel_embeddings})

        if self.pretrained_embeddings_entity != '':
            logger.info('Using pretrained entity embeddings.')
            pretrained_entities = np.load(self.pretrained_embeddings_entity)
            with open(self.pretrained_entity_to_id, 'r') as f:
                entity_to_id = json.load(f)
            ent_embeddings = sess.run(self.agent.entity_lookup_table)
            for entity, idx in entity_to_id.items():
                ent_embeddings[self.entity_vocab[entity]] = pretrained_entities[idx]
            sess.run(self.agent.entity_embedding_init,
                     feed_dict={self.agent.entity_embedding_placeholder: ent_embeddings})

    def bp(self, cost):
        self.baseline.update(tf.reduce_mean(input_tensor=self.cum_discounted_rewards))
        tvars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(ys=cost, xs=tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):   # See https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op

    def calc_cum_discounted_rewards(self, rewards):
        running_add = np.zeros([rewards.shape[0]])
        cum_disc_rewards = np.zeros([rewards.shape[0], self.path_length])
        cum_disc_rewards[:, self.path_length - 1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_rewards[:, t]
            cum_disc_rewards[:, t] = running_add
        return cum_disc_rewards

    def io_setup(self):
        fetches = self.per_example_loss + self.per_example_logits + self.actions_idx + [self.loss_op] + [self.dummy]
        feeds = self.candidate_relation_sequence + self.candidate_entity_sequence + self.entity_sequence + \
                [self.query_relations] + [self.range_arr] + [self.cum_discounted_rewards]

        feed_dict = [{} for _ in range(self.path_length)]
        feed_dict[0][self.query_relations] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size * self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None
        return fetches, feeds, feed_dict

    def beam_search(self, i, test_scores, beam_probs, temp_batch_size, states, agent_mem):
        k = self.test_rollouts
        new_scores = test_scores + beam_probs   # [k * B, max_number_actions]
        if i == 0:
            best_idx = np.argsort(new_scores)   # [k * B, max_number_actions]
            best_idx = best_idx[:, -k:]   # [k * B, k]
            ranged_idx = np.tile([b for b in range(k)], temp_batch_size)  # [k *B]
            best_idx = best_idx[np.arange(k * temp_batch_size), ranged_idx]   # [k * B]
        else:
            best_idx = self.top_k(new_scores, k)

        y = best_idx // self.max_num_actions
        x = best_idx % self.max_num_actions
        y += np.repeat([b * k for b in range(temp_batch_size)], k)
        states['current_entities'] = states['current_entities'][y]
        states['next_relations'] = states['next_relations'][y]
        states['next_entities'] = states['next_entities'][y]
        agent_mem = agent_mem[:, :, y, :]
        test_actions_idx = x
        chosen_relations = states['next_relations'][np.arange(k * temp_batch_size), x]
        beam_probs = new_scores[y, x].reshape((-1, 1))
        return chosen_relations, test_actions_idx, states, agent_mem, beam_probs, y

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, k * max_num_actions]
        best_idx = np.argsort(scores)
        best_idx = best_idx[:, -k:]
        return best_idx.reshape(-1)

    def paths_and_rules_stats(self, b, sorted_idx, qr, ce, end_e, test_rule_count_body, test_rule_count,
                              num_query_with_rules, num_query_with_rules_correct):
        rule_in_path = False
        is_correct = False
        answer_pos_rule = None
        pos_rule = 0
        seen_rule = set()
        for r in sorted_idx[b]:
            argument_temp = self.get_argument(b, r)
            key_temp = ' '.join(argument_temp[::2])
            self.paths_stats(argument_temp, key_temp, qr, end_e)
            body, obj = prepare_argument(argument_temp)
            rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule, test_rule_count_body, test_rule_count = \
                self.rules_stats(b, r, qr, body, obj, ce, end_e, key_temp, test_rule_count_body, test_rule_count,
                                 rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule)
        if rule_in_path:
            num_query_with_rules[0] += 1
            if qr[0] == '_':
                num_query_with_rules[1] += 1
            else:
                num_query_with_rules[2] += 1
            if is_correct:
                num_query_with_rules_correct[0] += 1
                if qr[0] == '_':
                    num_query_with_rules_correct[1] += 1
                else:
                    num_query_with_rules_correct[2] += 1
        return num_query_with_rules, num_query_with_rules_correct, answer_pos_rule, test_rule_count_body, \
               test_rule_count

    def get_argument(self, b, r):
        idx = b * self.test_rollouts + r
        argument_temp = [None] * (2 * len(self.relation_trajectory))
        argument_temp[::2] = [str(self.rev_relation_vocab[re[idx]]) for re in self.relation_trajectory]
        argument_temp[1::2] = [str(self.rev_entity_vocab[e[idx]]) for e in self.entity_trajectory][1:]
        return argument_temp

    def paths_stats(self, argument_temp, key_temp, qr, end_e):
        if qr in self.paths_body:
            if key_temp in self.paths_body[qr]:
                self.paths_body[qr][key_temp]['occurrences'] += 1
                if argument_temp[-1] == end_e:
                    self.paths_body[qr][key_temp]['correct_entities'] += 1
            else:
                self.paths_body[qr][key_temp] = {}
                self.paths_body[qr][key_temp]['relation'] = qr
                self.paths_body[qr][key_temp]['occurrences'] = 1
                if argument_temp[-1] == end_e:
                    self.paths_body[qr][key_temp]['correct_entities'] = 1
                else:
                    self.paths_body[qr][key_temp]['correct_entities'] = 0
        else:
            self.paths_body[qr] = {}
            self.paths_body[qr][key_temp] = {}
            self.paths_body[qr][key_temp]['relation'] = qr
            self.paths_body[qr][key_temp]['occurrences'] = 1
            if argument_temp[-1] == end_e:
                self.paths_body[qr][key_temp]['correct_entities'] = 1
            else:
                self.paths_body[qr][key_temp]['correct_entities'] = 0

    def rules_stats(self, b, r, qr, body, obj, ce, end_e, key_temp, test_rule_count_body, test_rule_count, rule_in_path,
                    is_correct, answer_pos_rule, pos_rule, seen_rule):
        rule_applied = False
        if qr in self.rule_list:
            rel_rules = self.rule_list[qr]
            for j in range(len(rel_rules)):
                if check_rule(body, obj, end_e, rel_rules[j], only_body=True):
                    rule_applied = True
                    rule_in_path = True
                    test_rule_count_body[0] += 1
                    if qr[0] == '_':
                        test_rule_count_body[1] += 1
                    else:
                        test_rule_count_body[2] += 1
                    if check_rule(body, obj, end_e, rel_rules[j], only_body=False):
                        is_correct = True
                        test_rule_count[0] += 1
                        if qr[0] == '_':
                            test_rule_count[1] += 1
                        else:
                            test_rule_count[2] += 1
                        if answer_pos_rule is None:
                            answer_pos_rule = pos_rule
                    break
        if (ce[b, r] not in seen_rule) and rule_applied:
            seen_rule.add(ce[b, r])
            pos_rule += 1
        if rule_applied:
            self.paths_body[qr][key_temp]['is_rule'] = 'yes'
        else:
            self.paths_body[qr][key_temp]['is_rule'] = 'no'
        return rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule, test_rule_count_body, test_rule_count

    def get_answer_pos(self, b, sorted_idx, rewards_reshape, ce):
        answer_pos = None
        pos = 0
        seen = set()
        if self.pool == 'max':
            for r in sorted_idx[b]:
                if rewards_reshape[b, r] == self.positive_reward:
                    answer_pos = pos
                    break
                if ce[b, r] not in seen:
                    seen.add(ce[b, r])
                    pos += 1
        if self.pool == 'sum':
            scores = defaultdict(list)
            answer = ''
            for r in sorted_idx[b]:
                scores[ce[b, r]].append(self.log_probs[b, r])
                if rewards_reshape[b, r] == self.positive_reward:
                    answer = ce[b, r]
            final_scores = defaultdict(float)
            for e in scores:
                final_scores[e] = lse(scores[e])
            sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
            if answer in sorted_answers:
                answer_pos = sorted_answers.index(answer)
            else:
                answer_pos = None
        return answer_pos

    def calculate_query_metrics(self, query_metrics, answer_pos):
        if answer_pos is not None:
            query_metrics[5] += 1.0 / (answer_pos + 1)
            if answer_pos < 20:
                query_metrics[4] += 1
                if answer_pos < 10:
                    query_metrics[3] += 1
                    if answer_pos < 5:
                        query_metrics[2] += 1
                        if answer_pos < 3:
                            query_metrics[1] += 1
                            if answer_pos < 1:
                                query_metrics[0] += 1
        return query_metrics

    def add_paths(self, b, sorted_idx, qr, start_e, se, ce, end_e, answer_pos, answers, rewards):
        self.paths[str(qr)].append(str(start_e) + '\t' + str(end_e) + '\n')
        self.paths[str(qr)].append('Reward:' + str(1 if (answer_pos is not None) and (answer_pos < 10) else 0) + '\n')
        for r in sorted_idx[b]:
            rev = -1
            idx = b * self.test_rollouts + r
            if rewards[idx] == self.positive_reward:
                rev = 1
            answers.append(self.rev_entity_vocab[se[b, r]] + '\t' + self.rev_entity_vocab[ce[b, r]] + '\t' +
                           str(self.log_probs[b, r]) + '\n')
            self.paths[str(qr)].append('\t'.join([str(self.rev_entity_vocab[e[idx]]) for e in self.entity_trajectory])
                                       + '\n' + '\t'.join([str(self.rev_relation_vocab[re[idx]]) for
                                                           re in self.relation_trajectory]) + '\n' +
                                       str(rev) + '\n' + str(self.log_probs[b, r]) + '\n___' + '\n')
        self.paths[str(qr)].append('#####################\n')
        return answers

    def write_paths_file(self, answers):
        for q in self.paths:
            j = q.replace('/', '-')
            with codecs.open(self.paths_log + '_' + j, 'a', 'utf-8') as pos_file:
                for p in self.paths[q]:
                    pos_file.write(p)
        with open(self.paths_log + 'answers', 'w') as answer_file:
            for a in answers:
                answer_file.write(a)

    def write_paths_summary(self):
        with open(self.output_dir + 'paths_body.json', 'w') as path_file:
            path_file.write('path\toccurrences\tcorrect_entities\tis_rule\trelation\n')
            path_file.write('####################\n')
            for qr in self.paths_body.keys():
                paths_body_sorted = sorted(self.paths_body[qr], key=lambda x: self.paths_body[qr][x]['occurrences'],
                                           reverse=True)
                for p in paths_body_sorted:
                    path_file.write(p + '\t' + str(self.paths_body[qr][p]['occurrences']) + '\t' +
                                    str(self.paths_body[qr][p]['correct_entities']) + '\t' +
                                    self.paths_body[qr][p]['is_rule'] + '\t' +
                                    self.paths_body[qr][p]['relation'] + '\n')
                path_file.write('####################\n')

    def write_scores_file(self, scores_file, final_metrics, final_metrics_rule, final_metrics_head,
                          final_metrics_rule_head, final_metrics_tail, final_metrics_rule_tail, test_rule_count_body,
                          test_rule_count, num_query_with_rules, num_query_with_rules_correct, total_examples):
        metrics = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@20', 'MRR']
        metrics_rule = ['Hits@1_rule', 'Hits@3_rule', 'Hits@5_rule', 'Hits@10_rule', 'Hits@20_rule', 'MRR_rule']
        ranking = ['Both:', 'Head:', 'Tail:']
        num_examples = [total_examples, total_examples / 2, total_examples / 2]
        all_results = [[final_metrics, final_metrics_rule],
                       [final_metrics_head, final_metrics_rule_head],
                       [final_metrics_tail, final_metrics_rule_tail]]
        for j in range(len(ranking)):
            scores_file.write(ranking[j])
            scores_file.write('\n')
            for i in range(len(metrics)):
                scores_file.write(metrics[i] + ': {0:7.4f}'.format(all_results[j][0][i]))
                scores_file.write('\n')
            for i in range(len(metrics_rule)):
                scores_file.write(metrics_rule[i] + ': {0:7.4f}'.format(all_results[j][1][i]))
                scores_file.write('\n')
            scores_file.write('Rule count body: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count_body[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count_body[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Rule count correct: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules[j]), int(num_examples[j]), num_query_with_rules[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule and correct: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules_correct[j]), int(num_examples[j]), num_query_with_rules_correct[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('\n')

    def train(self, sess):
        fetches, feeds, feed_dict = self.io_setup()
        train_loss = 0.0
        self.batch_counter = 0

        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relations] = episode.get_query_relations()
            states = episode.get_states()

            arguments = []
            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = states['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = states['next_entities']
                feed_dict[i][self.entity_sequence[i]] = states['current_entities']
                per_example_loss, per_example_logits, actions_idx = sess.partial_run(
                    h, [self.per_example_loss[i], self.per_example_logits[i], self.actions_idx[i]],
                    feed_dict=feed_dict[i])

                rel = np.copy(states['next_relations'][np.arange(states['next_relations'].shape[0]), actions_idx])
                ent = np.copy(states['next_entities'][np.arange(states['next_entities'].shape[0]), actions_idx])
                rel_string = np.array([self.rev_relation_vocab[x] for x in rel])
                ent_string = np.array([self.rev_entity_vocab[x] for x in ent])
                arguments.append(rel_string)
                arguments.append(ent_string)
                states = episode(actions_idx)

            query_rel_string = np.array([self.rev_relation_vocab[x] for x in episode.get_query_relations()])
            obj_string = np.array([self.rev_entity_vocab[x] for x in episode.get_query_objects()])

            rewards = episode.get_rewards()
            rewards, rule_count, rule_count_body = modify_rewards(self.rule_list, arguments, query_rel_string,
                                                                  obj_string, self.rule_base_reward, rewards,
                                                                  self.only_body)
            cum_discounted_rewards = self.calc_cum_discounted_rewards(rewards)

            # Backpropagation
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_rewards: cum_discounted_rewards})

            # Print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            num_hits = np.sum(rewards > 0)
            avg_reward = np.mean(rewards)
            rewards_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))
            rewards_reshape = np.sum(rewards_reshape, axis=1)
            num_ep_correct = np.sum(rewards_reshape > 0)
            if np.isnan(train_loss):
                raise ArithmeticError('Error in computing loss.')

            logger.info('batch_counter: {0:4d}, num_hits: {1:7d}, avg_reward: {2:6.4f}, num_ep_correct: {3:4d}, '
                        'avg_ep_correct: {4:6.4f}, train_loss: {5:6.4f}'.
                        format(self.batch_counter, num_hits, avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size), train_loss))
            logger.info('rule_count_body: {0}/{1} = {2:6.4f}'.format(
                rule_count_body, self.batch_size * self.num_rollouts,
                rule_count_body / (self.batch_size * self.num_rollouts)))
            logger.info('rule_count_correct: {0}/{1} = {2:6.4f}'.format(
                rule_count, self.batch_size * self.num_rollouts,
                rule_count / (self.batch_size * self.num_rollouts)))

            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + 'scores.txt', 'a') as score_file:
                    score_file.write('Scores for iteration ' + str(self.batch_counter) + '\n')
                paths_log_dir = self.output_dir + str(self.batch_counter) + '/'
                os.makedirs(paths_log_dir)
                self.paths_log = paths_log_dir + 'paths'
                self.test(sess)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            gc.collect()

            if self.early_stopping:
                break
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, print_paths=False, save_model=True, beam=True):
        self.paths = defaultdict(list)
        self.paths_body = dict()
        batch_counter = 0
        answers = []
        feed_dict = {}
        # For the calculation of the rankings ['Both:', 'Head:', 'Tail:']
        test_rule_count_body = np.zeros(3)
        test_rule_count = np.zeros(3)
        num_query_with_rules = np.zeros(3)
        num_query_with_rules_correct = np.zeros(3)
        # For the calculation of the metrics [h@1, h@3, h@5, h@10, h@20, MRR]
        final_metrics = np.zeros(6)
        final_metrics_rule = np.zeros(6)
        final_metrics_head = np.zeros(6)
        final_metrics_rule_head = np.zeros(6)
        final_metrics_tail = np.zeros(6)
        final_metrics_rule_tail = np.zeros(6)
        total_examples = self.test_environment.total_no_examples

        if self.test_environment == self.test_test_environment:
            scores_extended_file = open(self.output_dir + 'scores_extended.txt', 'w')
            scores_extended_file.write('Test scores (reciprocal ranks) for all test triples \n')

        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1
            temp_batch_size = episode.no_examples
            self.qrs = episode.get_query_relations()
            feed_dict[self.query_relations] = episode.get_query_relations()
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            states = episode.get_states()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size * self.test_rollouts, mem[3])).astype('float32')
            previous_relations = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * \
                                 self.relation_vocab['DUMMY_START_RELATION']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            self.log_probs = np.zeros((temp_batch_size * self.test_rollouts, )) * 1.0
            self.entity_trajectory = []
            self.relation_trajectory = []

            for i in range(self.path_length):
                feed_dict[self.next_relations] = states['next_relations']
                feed_dict[self.next_entities] = states['next_entities']
                feed_dict[self.current_entities] = states['current_entities']
                feed_dict[self.prev_states] = agent_mem
                feed_dict[self.prev_relations] = previous_relations

                loss, test_scores, agent_mem, test_actions_idx, chosen_relations = sess.run(
                    [self.test_loss, self.test_logits, self.test_state, self.test_actions_idx, self.chosen_relations],
                    feed_dict=feed_dict)

                if beam:
                    chosen_relations, test_actions_idx, states, agent_mem, beam_probs, y = \
                        self.beam_search(i, test_scores, beam_probs, temp_batch_size, states, agent_mem)
                    for j in range(i):
                        self.entity_trajectory[j] = self.entity_trajectory[j][y]
                        self.relation_trajectory[j] = self.relation_trajectory[j][y]

                previous_relations = chosen_relations
                self.entity_trajectory.append(states['current_entities'])
                self.relation_trajectory.append(chosen_relations)
                states = episode(test_actions_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_actions_idx]

            if beam:
                self.log_probs = beam_probs

            self.entity_trajectory.append(states['current_entities'])

            # Ask environment for final reward
            rewards = episode.get_rewards()
            rewards_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_idx = np.argsort(-self.log_probs)

            query_metrics = np.zeros(6)
            query_metrics_rule = np.zeros(6)
            query_metrics_head = np.zeros(6)
            query_metrics_rule_head = np.zeros(6)
            query_metrics_tail = np.zeros(6)
            query_metrics_rule_tail = np.zeros(6)

            ce = episode.states['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

            for b in range(temp_batch_size):
                qr = self.train_environment.grapher.rev_relation_vocab[self.qrs[b * self.test_rollouts]]
                start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]

                num_query_with_rules, num_query_with_rules_correct, answer_pos_rule, test_rule_count_body, \
                test_rule_count = self.paths_and_rules_stats(b, sorted_idx, qr, ce, end_e, test_rule_count_body,
                                                             test_rule_count, num_query_with_rules,
                                                             num_query_with_rules_correct)
                answer_pos = self.get_answer_pos(b, sorted_idx, rewards_reshape, ce)
                query_metrics = self.calculate_query_metrics(query_metrics, answer_pos)
                query_metrics_rule = self.calculate_query_metrics(query_metrics_rule, answer_pos_rule)
                if qr[0] == '_':   # Inverse triple
                    query_metrics_head = self.calculate_query_metrics(query_metrics_head, answer_pos)
                    query_metrics_rule_head = self.calculate_query_metrics(query_metrics_rule_head, answer_pos_rule)
                else:
                    query_metrics_tail = self.calculate_query_metrics(query_metrics_tail, answer_pos)
                    query_metrics_rule_tail = self.calculate_query_metrics(query_metrics_rule_tail, answer_pos_rule)
                if print_paths:
                    answers = self.add_paths(b, sorted_idx, qr, start_e, se, ce, end_e, answer_pos, answers, rewards)

                if self.test_environment == self.test_test_environment:
                    scores_extended_file.write(start_e + '\t' + qr + '\t' + end_e + '\n')
                    if answer_pos is None:
                        scores_extended_file.write('Reciprocal rank is None \n')
                    else:
                        scores_extended_file.write('Reciprocal rank: {0:5.4f}'.format(1.0 / (answer_pos + 1)))
                        scores_extended_file.write('\n')
                    if answer_pos_rule is None:
                        scores_extended_file.write('Reciprocal rank (rule) is None \n')
                    else:
                        scores_extended_file.write('Reciprocal rank (rule): {0:5.4f}'.format(1.0 /
                                                                                             (answer_pos_rule + 1)))
                        scores_extended_file.write('\n')

            final_metrics += query_metrics
            final_metrics_rule += query_metrics_rule
            final_metrics_head += query_metrics_head
            final_metrics_rule_head += query_metrics_rule_head
            final_metrics_tail += query_metrics_tail
            final_metrics_rule_tail += query_metrics_rule_tail

        final_metrics /= total_examples
        final_metrics_rule /= total_examples
        final_metrics_head /= total_examples / 2
        final_metrics_rule_head /= total_examples / 2
        final_metrics_tail /= total_examples / 2
        final_metrics_rule_tail /= total_examples / 2

        if save_model:
            if final_metrics[-1] > self.best_metric:
                self.best_metric = final_metrics[-1]
                self.model_saver.save(sess, self.model_dir + 'model.ckpt')
                self.current_patience = self.patience
            elif self.best_metric >= final_metrics[-1]:
                self.current_patience -= 1
                if self.current_patience == 0:
                    self.early_stopping = True

        self.write_paths_summary()
        if print_paths:
            logger.info('Printing paths at {}'.format(self.output_dir + 'test_beam/'))
            self.write_paths_file(answers)

        with open(self.output_dir + 'scores.txt', 'a') as scores_file:
            self.write_scores_file(scores_file, final_metrics, final_metrics_rule, final_metrics_head,
                                   final_metrics_rule_head, final_metrics_tail, final_metrics_rule_tail,
                                   test_rule_count_body, test_rule_count, num_query_with_rules,
                                   num_query_with_rules_correct, total_examples)

        metrics = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@20', 'MRR']
        for i in range(len(metrics)):
            logger.info(metrics[i] + ': {0:7.4f}'.format(final_metrics[i]))
        metrics_rule = ['Hits@1_rule', 'Hits@3_rule', 'Hits@5_rule', 'Hits@10_rule', 'Hits@20_rule', 'MRR_rule']
        for i in range(len(metrics_rule)):
            logger.info(metrics_rule[i] + ': {0:7.4f}'.format(final_metrics_rule[i]))


def create_output_and_model_dir(params, mode):
    current_time = datetime.datetime.now()
    current_time = current_time.strftime('%d%b%y_%H%M%S')
    if mode == 'test':
        params['output_dir'] = params['base_output_dir'] + str(current_time) + '_TEST' + \
                               '_p' + str(params['path_length']) + '_r' + str(params['rule_base_reward']) + \
                               '_e' + str(params['embedding_size']) + '_h' + str(params['hidden_size']) + \
                               '_L' + str(params['LSTM_layers']) + '_l' + str(params['learning_rate']) + \
                               '_o' + str(params['only_body']) + '/'
        os.makedirs(params['output_dir'])
    else:
        params['output_dir'] = params['base_output_dir'] + str(current_time) + \
                               '_p' + str(params['path_length']) + '_r' + str(params['rule_base_reward']) + \
                               '_e' + str(params['embedding_size']) + '_h' + str(params['hidden_size']) + \
                               '_L' + str(params['LSTM_layers']) + '_l' + str(params['learning_rate']) + \
                               '_o' + str(params['only_body']) + '/'
        params['model_dir'] = params['output_dir'] + 'model/'
        os.makedirs(params['output_dir'])
        os.makedirs(params['model_dir'])
    return params


def initialize_setting(params, relation_vocab, entity_vocab, mode=''):
    params = create_output_and_model_dir(params, mode)
    params.pop('relation_vocab', None)
    params.pop('entity_vocab', None)
    with open(params['output_dir'] + 'config.txt', 'w') as out:
        pprint(params, stream=out)
    maxLen = max([len(k) for k in params.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(params.items()):
        print(fmtString % keyPair)
    params['relation_vocab'] = relation_vocab
    params['entity_vocab'] = entity_vocab
    return params


if __name__ == '__main__':
    options = read_options()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %H:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = None
    logger.info('Reading vocab files...')
    vocab_dir = options['input_dir'] + 'vocab/'
    relation_vocab = json.load(open(vocab_dir + 'relation_vocab.json'))
    entity_vocab = json.load(open(vocab_dir + 'entity_vocab.json'))
    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    if not options['load_model']:
        for k, v in options.items():
            if not isinstance(v, list):
                options[k] = [v]
        best_permutation = None
        best_metric = -1
        for permutation in ParameterGrid(options):
            permutation = initialize_setting(permutation, relation_vocab, entity_vocab)
            logger.removeHandler(logfile)
            logfile = logging.FileHandler(permutation['output_dir'] + 'log.txt', 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            # Training
            trainer = Trainer(permutation)
            with tf.compat.v1.Session(config=config) as sess:
                sess.run(trainer.initialize())
                trainer.initialize_pretrained_embeddings(sess=sess)
                trainer.train(sess)

            if (best_permutation is None) or (trainer.best_metric > best_metric):
                best_metric = trainer.best_metric
                best_permutation = permutation
            tf.compat.v1.reset_default_graph()

        # Testing on test set with best model
        best_permutation['old_output_dir'] = best_permutation['output_dir']
        best_permutation = initialize_setting(best_permutation, relation_vocab, entity_vocab, mode='test')
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(best_permutation['output_dir'] + 'log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        trainer = Trainer(best_permutation)
        model_path = best_permutation['old_output_dir'] + 'model/model.ckpt'
        output_dir = best_permutation['output_dir']

        with tf.compat.v1.Session(config=config) as sess:
            trainer.initialize(model_path, sess)
            os.makedirs(output_dir + 'test_beam/')
            trainer.paths_log = output_dir + 'test_beam/paths'
            with open(output_dir + 'scores.txt', 'a') as scores_file:
                scores_file.write('Test (beam) scores with best model from ' + model_path + '\n')
            trainer.test_environment = trainer.test_test_environment
            trainer.test(sess, print_paths=True, save_model=False)

    else:
        for k, v in options.items():
            if isinstance(v, list):
                if len(v) == 1:
                    options[k] = v[0]
                else:
                    raise ValueError('Parameter {} has more than one value in the config file.'.format(k))
        logger.info('Skipping training...')
        model_path = options['model_load_path']
        logger.info('Loading model from {}'.format(model_path))
        options = initialize_setting(options, relation_vocab, entity_vocab, mode='test')
        output_dir = options['output_dir']
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(output_dir + 'log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        trainer = Trainer(options)

        with tf.compat.v1.Session(config=config) as sess:
            trainer.initialize(restore=model_path, sess=sess)
            os.makedirs(output_dir + 'test_beam/')
            trainer.paths_log = output_dir + 'test_beam/paths'
            with open(output_dir + 'scores.txt', 'a') as scores_file:
                scores_file.write('Test (beam) scores with best model from ' + model_path + '\n')
            trainer.test_environment = trainer.test_test_environment
            trainer.test(sess, print_paths=True, save_model=False)
