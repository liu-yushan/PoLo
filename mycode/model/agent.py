import numpy as np
import tensorflow as tf


class Agent(object):
    def __init__(self, params):
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.LSTM_Layers = params['LSTM_layers']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_labels = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
            self.entity_initializer = tf.keras.initializers.GlorotUniform()
        else:
            self.m = 2
            self.entity_initializer = tf.compat.v1.zeros_initializer()

        with tf.compat.v1.variable_scope('action_lookup_table'):
            self.action_embedding_placeholder = tf.compat.v1.placeholder(
                tf.float32, [self.action_vocab_size, 2 * self.embedding_size])
            self.relation_lookup_table = tf.compat.v1.get_variable(
                'relation_lookup_table', shape=[self.action_vocab_size, 2 * self.embedding_size],
                dtype=tf.float32, initializer=tf.keras.initializers.GlorotUniform(), trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.compat.v1.variable_scope('entity_lookup_table'):
            self.entity_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.compat.v1.get_variable(
                'entity_lookup_table', shape=[self.entity_vocab_size, 2 * self.embedding_size],
                dtype=tf.float32, initializer=self.entity_initializer, trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.compat.v1.variable_scope('policy_step'):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True,
                                                     state_is_tuple=True))
            self.policy_step = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    def get_mem_shape(self):
        return self.LSTM_Layers, 2, None, self.m * self.hidden_size

    def policy_MLP(self, state):
        with tf.compat.v1.variable_scope('MLP_for_policy'):
            hidden = tf.compat.v1.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.compat.v1.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(self, next_relations, next_entities):
        with tf.compat.v1.variable_scope('lookup_table_edge_encoder'):
            relation_embedding = tf.compat.v1.nn.embedding_lookup(params=self.relation_lookup_table, ids=next_relations)
            entity_embedding = tf.compat.v1.nn.embedding_lookup(params=self.entity_lookup_table, ids=next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, current_entities, prev_states, prev_relations, query_embeddings,
             range_arr):
        prev_action_embeddings = self.action_encoder(prev_relations, current_entities)

        # One step of RNN
        output, new_states = self.policy_step(prev_action_embeddings, prev_states)

        # Get state vector
        prev_entities = tf.compat.v1.nn.embedding_lookup(params=self.entity_lookup_table, ids=current_entities)
        if self.use_entity_embeddings:
            states = tf.concat([output, prev_entities], axis=-1)
        else:
            states = output
        state_query_concat = tf.concat([states, query_embeddings], axis=-1)
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)

        # MLP for policy
        output = self.policy_MLP(state_query_concat)
        output_expanded = tf.expand_dims(output, axis=1)
        prelim_scores = tf.reduce_sum(input_tensor=tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions
        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD
        mask = tf.equal(next_relations, comparison_tensor)
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0
        scores = tf.compat.v1.where(mask, dummy_scores, prelim_scores)

        # Sample action
        actions = tf.cast(tf.random.categorical(logits=scores, num_samples=1), dtype=tf.int32)

        # Loss
        labels_actions = tf.squeeze(actions, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels_actions)

        actions_idx = tf.squeeze(actions)
        chosen_relations = tf.gather_nd(next_relations, tf.transpose(a=tf.stack([range_arr, actions_idx])))

        return loss, tf.nn.log_softmax(scores), new_states, actions_idx, chosen_relations

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 query_relations, range_arr, path_length):
        query_embeddings = tf.compat.v1.nn.embedding_lookup(params=self.relation_lookup_table, ids=query_relations)
        states = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        prev_relations = self.dummy_start_labels
        all_loss = []
        all_logits = []
        actions_idx = []

        with tf.compat.v1.variable_scope('policy_steps_unroll') as scope:
            for t in range(path_length):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]
                loss, logits, new_states, idx, chosen_relations = self.step(
                    next_possible_relations, next_possible_entities, current_entities_t, states, prev_relations,
                    query_embeddings, range_arr)
                all_loss.append(loss)
                all_logits.append(logits)
                actions_idx.append(idx)
                prev_relations = chosen_relations

        return all_loss, all_logits, actions_idx
