#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
from mlx.python.util.graph_state_helper import graph_state_helper
import mlx.python.tf as tfmlx
import tensorflow as tf
global g_mlx_feature_names  # All sparse feature name list
global g_mlx_embed_names  # All sparse mlx emb name list
global g_mlx_embed_mapping  # All sparse mlx emb/feature name, emb op map
global g_mlx_seq_feature_names
global g_mlx_embed_seq_mapping  # All sparse mlx emb/feature name, emb seq op map
global g_mlx_embed_mask_mapping  # All sparse mlx emb/feature name, emb mask op map
global g_my_name_collections
global g_seq_embed_mapping
global g_seq_embed_names

extra_graph_states = [
    graph_state_helper('TRAIN'),
    graph_state_helper('EVALUATE'),
    graph_state_helper('PREDICT', 'default'),
    graph_state_helper('PREDICT', 'rank_online_serving')
]


def add_input_layer(input_src, scope_name='default_input'):
    with tf.variable_scope(scope_name):
        input_dim = int(input_src.shape[1])
        weight_1 = tf.get_variable(
            "input_w_1", [1, input_dim],
            initializer=tfmlx.xavier_initializer(mode='COUNT_COL')
        )
        weight_2 = tf.get_variable(
            "input_w_2", [1, input_dim],
            initializer=tfmlx.xavier_initializer(mode='COUNT_COL')
        )
        zeros = tf.cast(tf.less(input_src, 1e-6), tf.float32)
        output = tf.multiply(input_src, weight_1) + tf.multiply(zeros, weight_2)
        return output
def add_fc(input_var, units, activation=None, name='fc'):
    assert isinstance(units, int), "units should be int"
    with tf.variable_scope(name):
        input_dim = int(input_var.shape[1])
        var_w = tf.get_variable(
            'w', [units, input_dim],
            initializer=tfmlx.xavier_initializer(mode='COUNT_COL')
        )
        var_b = tf.get_variable(
            'b', [1, units],
            initializer=tf.zeros_initializer()
        )
        h = tf.matmul(input_var, tf.transpose(var_w)) + var_b
        if type(activation) is str:
            return activation_layer.get(activation)(h)
        elif type(activation) is callable:
            return activation(h)
        return h
def activation_layer(activation, prefix, is_training):
    if isinstance(activation, str):
        if activation == 'prelu':
            return None
        elif activation == 'dice':
            return Dice(prefix, is_training)
        elif activation == 'evonorm':
            return EvoNorm(prefix, is_training)
        elif activation == 'evonormnew':
            return EvoNormNew(prefix, is_training)
        elif activation == 'silu':
            return Silu(prefix, is_training)
        elif activation == 'evonormseq':
            return EvoNormSeq(prefix, is_training)
        return tf.keras.layers.Activation(activation)
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % activation)

class Silu(tf.keras.layers.Layer):
    def __init__(self, prefix, is_training, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        super(Silu, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = tf.get_variable(
            name='{}_silu_gama'.format(self.prefix),
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        super(Silu, self).build(input_shape)
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.gamma * inputs)

class Dice(tf.keras.layers.Layer):
    def __init__(self, prefix, is_training, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        super(Dice, self).__init__(**kwargs)
    def build(self, input_shape):
        self.alphas = tf.get_variable(name='dice_alpha_' + self.prefix, shape=input_shape[1:],
                                      initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        super(Dice, self).build(input_shape)
    def call(self, inputs):
        inputs_normed = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            epsilon=1e-9,
            center=True,
            scale=True,
            training=self.is_training)
        p_x = tf.sigmoid(inputs_normed)
        # f(x) = p(x)*x+(1-p(x))*alpha*x
        return p_x * inputs + (1.0 - p_x) * self.alphas * inputs
class EvoNorm(tf.keras.layers.Layer):
    def __init__(self, prefix, is_training, rank=1, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        self.rank = rank
        super(EvoNorm, self).__init__(**kwargs)
    def build(self, input_shape):
        self.beta = tf.get_variable(
            name='{}_EvoNorm_beta'.format(self.prefix),
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        self.gamma = tf.get_variable(
            name='{}_EvoNorm_gama'.format(self.prefix),
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        self.v = tf.get_variable(
            name='{}_EvoNorm_v'.format(self.prefix),
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        super(EvoNorm, self).build(input_shape)
    def call(self, inputs):
        epsilon = 0.000000001
        _, variance = tf.nn.moments(inputs, 1, keep_dims=True)
        std = tf.sqrt(variance + epsilon)
        # std = tf.broadcast_to(std, tf.shape(inputs))  # 用broadcast会出现 no gradient defined for operation broadcast_to ...
        std = tf.tile(std, [1, inputs.shape[1]])
        # std = tf.broadcast_to(std, list(inputs.shape))
        inputs = inputs * tf.nn.sigmoid(self.v * inputs) / std
        return inputs * self.gamma + self.beta

class EvoNormNew(tf.keras.layers.Layer):
    def __init__(self, prefix, is_training, rank=1, last=False, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        self.rank = rank
        self.last = last
        super(EvoNormNew, self).__init__(**kwargs)
    def build(self, input_shape):
        shape_i = self.rank
        shape_j = shape_i + 1
        if self.last:
            shape_i = -1
            shape_j = 3
        self.beta = tf.get_variable(
            name='{}_EvoNorm_beta'.format(self.prefix),
            shape=input_shape[shape_i:shape_j],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        self.gamma = tf.get_variable(
            name='{}_EvoNorm_gama'.format(self.prefix),
            shape=input_shape[shape_i:shape_j],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        self.v = tf.get_variable(
            name='{}_EvoNorm_v'.format(self.prefix),
            shape=input_shape[shape_i:shape_j],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        super(EvoNormNew, self).build(input_shape)
    def call(self, inputs):
        epsilon = 0.000000001
        inputs_new = inputs
        if not self.last:
            inputs_new = tf.transpose(inputs, perm=[0, 2, 1])
        _, variance = tf.nn.moments(inputs_new, -1, keep_dims=True)
        std = tf.sqrt(variance + epsilon)
        inputs_new = inputs_new * tf.nn.sigmoid(self.v * inputs_new) / std
        inputs_new = inputs_new * self.gamma + self.beta
        res = inputs_new
        if not self.last:
            res = tf.transpose(inputs_new, perm=[0, 2, 1])
        return res

class EvoNormSeq(tf.keras.layers.Layer):
    def __init__(self, prefix, is_training, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        super(EvoNormSeq, self).__init__(**kwargs)
    def build(self, input_shape):
        print("input_shape:zhangbo:" + str(input_shape))
        self.beta = tf.get_variable(
            name='{}_EvoNorm_beta'.format(self.prefix),
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        self.gamma = tf.get_variable(
            name='{}_EvoNorm_gama'.format(self.prefix),
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        self.v = tf.get_variable(
            name='{}_EvoNorm_v'.format(self.prefix),
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        super(EvoNormSeq, self).build(input_shape)
    def call(self, inputs):
        epsilon = 0.000000001
        _, variance = tf.nn.moments(inputs, axes=-1, keep_dims=True)
        std = tf.sqrt(variance + epsilon)
        # std = tf.broadcast_to(std, tf.shape(inputs))  # 用broadcast会出现 no gradient defined for operation broadcast_to ...
        # std = tf.tile(std, [1,inputs.shape[1]])
        # std = tf.broadcast_to(std, list(inputs.shape))
        inputs = inputs * tf.nn.sigmoid(self.v * inputs) / std
        return inputs * self.gamma + self.beta
class DNN(tf.keras.layers.Layer):
    def __init__(self, hidden_units, name, is_training=True, activation='relu', output_activation=None, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.output_activation = output_activation or self.activation
        self.is_training = is_training
        name = "{0}_{1}".format(name, "-".join(str(i) for i in hidden_units))
        super(DNN, self).__init__(name=name, **kwargs)
    def _get_fc_name(self, layer, unit):
        return 'fc{0}_{1}'.format(layer, unit)
    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        with tf.variable_scope(self.name):
            self.kernels = []
            self.bias = []
            self.activation_layers = []
            for i in range(len(self.hidden_units)):
                with tf.variable_scope(self._get_fc_name(i, self.hidden_units[i])):
                    kernel = tf.get_variable('kernel_' + str(i),
                                             [hidden_units[i], hidden_units[i + 1]],
                                             initializer=tfmlx.xavier_initializer(mode='COUNT_COL'))
                    b = tf.get_variable('bias_' + str(i),
                                        [hidden_units[i + 1]],
                                        initializer=tf.zeros_initializer())
                    if i == len(self.hidden_units) - 1:
                        active = activation_layer(self.output_activation, self.name + "_output", self.is_training)
                    else:
                        active = activation_layer(self.activation, self.name + str(i), self.is_training)
                    self.kernels.append(kernel)
                    self.bias.append(b)
                    self.activation_layers.append(active)
        super(DNN, self).build(input_shape)
    def call(self, inputs, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            with tf.name_scope(self._get_fc_name(i, self.hidden_units[i])):
                fc = tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0))
                fc = tf.nn.bias_add(fc, self.bias[i])
                deep_input = self.activation_layers[i](fc)
        return deep_input
class ContextualMultiHeadMulAttention2(tf.keras.layers.Layer):
    def __init__(self, prefix, dims, heads=1, activation='relu', output_activation=None, **kwargs):
        self.prefix = prefix
        self.dims = dims
        self.heads = heads
        self.activation = activation
        self.output_activation = output_activation or self.activation
        super(ContextualMultiHeadMulAttention2, self).__init__(**kwargs)
    def build(self, input_shape):
        self.key_shape = input_shape[0]
        self.query_dnns = [DNN(self.dims, self.prefix + "_dnn_" + str(i), activation=self.activation,
                               output_activation=self.output_activation)
                           for i in range(self.heads)]
        super(ContextualMultiHeadMulAttention2, self).build(input_shape)
    def call(self, inputs):
        keys = inputs[0]  # [?, seq_size, emb1]
        querys = inputs[1]  # [?, seq_size, emb3]
        contexts = inputs[2]  # [?, emb2]
        mask = inputs[3]  # [?, seq_size]
        mask = tf.expand_dims(mask, -1)  # [?, seq_size, 1]
        contexts = tf.keras.backend.repeat_elements(tf.expand_dims(contexts, 1), int(self.key_shape[1]),
                                                    1)  # [?, seq_size,emb2]
        querys = tf.concat([querys, contexts], 2)  # 把时间序列和复制版的上下文时间拼接起来，一起走dnn
        result = tf.concat([tf.reduce_sum(keys * self.query_dnns[i](querys) * mask, axis=1) for i in range(self.heads)],
                           axis=1)
        return result  # [?, heads*hidden_output]
class ContextualMultiHeadMulAttention(tf.keras.layers.Layer):
    def __init__(self, prefix, dims, heads=1, activation='relu', output_activation=None, **kwargs):
        self.prefix = prefix
        self.dims = dims
        self.heads = heads
        self.activation = activation
        self.output_activation = output_activation or self.activation
        super(ContextualMultiHeadMulAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.key_shape = input_shape[0]
        self.query_dnns = [DNN(self.dims, self.prefix + "_dnn_" + str(i), activation=self.activation,
                               output_activation=self.output_activation)
                           for i in range(self.heads)]
        super(ContextualMultiHeadMulAttention, self).build(input_shape)
    def call(self, inputs):
        keys = inputs[0]  # [?, seq_size, emb1]
        query = inputs[1]  # [[?, seq_size, emb3] *3]
        contexts = inputs[2]  # [?, seq_size, emb2]
        mask = inputs[3]  # [?, seq_size]
        mask = tf.expand_dims(mask, -1)  # [?, seq_size, 1]
        querys = tf.concat(query + [contexts], 2)  # 把时间序列和复制版的上下文时间拼接起来，一起走dnn
        result = tf.concat([tf.reduce_sum(keys * self.query_dnns[i](querys) * mask, axis=1) for i in range(self.heads)],
                           axis=1)
        return result  # [?, heads*hidden_output]
class CGC(tf.keras.layers.Layer):
    def __init__(self, n_experts, share_n_experts, n_tasks, hiddens, share_hiddens, is_training,
                 prefix='cgc', activation='relu',
                 **kwargs):
        self.n_experts = n_experts
        self.share_n_experts = share_n_experts
        self.n_tasks = n_tasks
        self.hiddens = hiddens
        self.share_hiddens = share_hiddens
        # self.gate_hiddens = gate_hiddens
        self.activation = activation
        # self.gate_activation = gate_activation
        self.prefix = prefix
        # self.gate_output_activation = gate_output_activation
        self.is_training = is_training
        super(CGC, self).__init__(name=prefix, **kwargs)
    def build(self, input_shape):
        if not self.n_experts == self.n_tasks:
            raise ValueError(
                'CGC layer n_experts:{0} should be equal to n_tasks:{1}'.format(str(self.n_experts), str(self.n_tasks))
            )
        self.expert_dnns = []
        self.share_expert_dnns = []
        # self.gate_dnns = []
        with tf.variable_scope(self.name):
            for i in range(self.n_experts):  # 2
                dnn = DNN(self.hiddens,
                          name='expert_dnn{0}'.format(i),
                          is_training=self.is_training,
                          activation=self.activation)
                self.expert_dnns.append(dnn)
                dnn.build(input_shape[0])
                dnn.built = True
            for i in range(self.share_n_experts):  # 2
                share_dnn = DNN(self.share_hiddens,
                                name='share_expert_dnn{0}'.format(i),
                                is_training=self.is_training,
                                activation=self.activation)
                self.share_expert_dnns.append(share_dnn)
                share_dnn.build(input_shape[0])
                share_dnn.built = True
            # for i in range(self.n_tasks):  # 2
            #     gate_dnn = DNN(self.gate_hiddens + [self.share_n_experts + 1],
            #                    name=self.prefix + '_gate_dnn{0}'.format(i),
            #                    is_training=self.is_training,
            #                    activation=self.gate_activation,
            #                    output_activation=self.gate_output_activation)
            #     self.gate_dnns.append(gate_dnn)
            #     gate_dnn.build(input_shape[1])
            #     gate_dnn.built = True
            # super(CGC, self).build(input_shape)
    def call(self, inputs):
        expert_input = inputs[0]
        # gate_input = inputs[1]
        expert_outputs = [dnn(expert_input) for dnn in self.expert_dnns]  # 2个[?,192]
        share_expert_outputs = [dnn(expert_input) for dnn in self.share_expert_dnns]  # 2个[?,192]
        # gate_outputs = [dnn(gate_input) for dnn in self.gate_dnns]  # 2个[?,2+1]
        final_outputs = []
        for i in range(2):  # 2  gate_output:
            task_output = tf.stack([expert_outputs[i]] + share_expert_outputs, axis=1)  # [?, 3, 192]
            task_expert_output = tf.reduce_sum(task_output,
                                               axis=1,
                                               name="{0}_out_{1}".format(self.prefix, i))
            final_outputs.append(task_expert_output)
        return final_outputs
conf = {
    "dense": {
        "dense_len": 151
    },
    "sparse": {
        "user_sparse_list": [
            "1  f_gender 2",
            "2  f_age 4",
            "3  f_constellation   4",
            "4  f_married   2",
            "7  f_job 4",
            "8  f_user_level   4",
            "10 f_cityid 6",
            "11 f_loc_cityid   6",
            "13 f_client_type  4",
            "14 f_reslution 4",
            "15 f_agent  4",
            "16 f_app_source   4",
            "17 f_app_version  4",
            "24 f_user_id   12",
            "25 f_userid0   2",
            "28 f_item_type_id 12",
            "29 f_dtype  4",
            "46 f_exp_item_30days_exposure_cnt_1 0",
            "75 f_weekday   36",
            "74 f_hour   36",
            "79 f_geohash5  8",
            "80 f_geohash   12",
            "84 f_isremote  2",
            "85 f_remote_type  2",
            "162   f_user_view_item_dtype_lists_6h  12    f_item_type_id",
            "170   f_user_view_class_lists_6h 8",
            "174   f_user_view_type_lists_6h  8",
            "178   f_user_view_cate_lists_6h  8",
            "202   f_user_order_brandid_lists_6h 12",
            "203   f_user_view_brandid_lists_6h  12    f_user_order_brandid_lists_6h",
            "233   f_user_view_item_dtype_lists_24h 12    f_item_type_id",
            "238   f_user_order_class_lists_24h  8  f_user_view_class_lists_6h",
            "239   f_user_order_type_lists_24h   8  f_user_view_type_lists_6h",
            "241   f_user_view_class_lists_24h   8  f_user_view_class_lists_6h",
            "245   f_user_view_type_lists_24h 8  f_user_view_type_lists_6h",
            "249   f_user_view_cate_lists_24h 8 f_user_view_cate_lists_6h",
            "273   f_user_order_brandid_lists_24h   12    f_user_order_brandid_lists_6h",
            "274   f_user_view_brandid_lists_24h 12    f_user_order_brandid_lists_6h",
            "304   f_user_view_item_dtype_lists  12    f_item_type_id",
            "308   f_user_order_item_dtype_lists 12    f_item_type_id",
            "309   f_user_order_class_lists   8  f_user_view_class_lists_6h",
            "310   f_user_order_type_lists 8  f_user_view_type_lists_6h",
            "311   f_user_order_cate_lists 8 f_user_view_cate_lists_6h",
            "312   f_user_view_class_lists 8  f_user_view_class_lists_6h",
            "316   f_user_view_type_lists  8  f_user_view_type_lists_6h",
            "320   f_user_view_cate_lists  8 f_user_view_cate_lists_6h",
            "344   f_user_order_brandid_lists 12    f_user_order_brandid_lists_6h",
            "345   f_user_view_brandid_lists  12    f_user_order_brandid_lists_6h",
            "92 f_user_1h_most_view_cate   8",
            "93 f_user_1h_most_view_type   8",
            "94 f_user_1h_most_view_class  8",
            "102   f_user_1h_most_two_view_cate  8    f_user_1h_most_view_cate",
            "103   f_user_1h_most_three_view_class  8    f_user_1h_most_view_class",
            "104   f_user_6h_most_view_type   8    f_user_1h_most_view_type",
            "107   f_user_6h_most_three_view_class_list   8    f_user_1h_most_view_class",
            "118   f_user_1w_most_three_view_class  8    f_user_1h_most_view_class",
            "119   f_1d_last_view_two_cates   8    f_user_1h_most_view_cate",
            "120   f_1d_last_view_three_class 8    f_user_1h_most_view_class",
            "121   f_3d_last_view_cates 8    f_user_1h_most_view_cate",
            "122   f_3d_last_view_two_class   8    f_user_1h_most_view_class",
            "123   f_3d_last_view_three_class 8    f_user_1h_most_view_class",
            "124   f_1w_last_view_cates 8    f_user_1h_most_view_cate",
            "125   f_1w_last_view_two_class   8    f_user_1h_most_view_class",
            "126   f_all_last_view_cates   8    f_user_1h_most_view_cate",
            "157   f_2h_last_order_types   8    f_user_1h_most_view_type",
            "158   f_2h_last_order_class   8    f_user_1h_most_view_class",
            "159   f_2h_last_order_two_cates  8    f_user_1h_most_view_cate",
            "160   f_12h_last_order_two_types 8    f_user_1h_most_view_type",
            "164   f_2d_last_order_three_class   8    f_user_1h_most_view_class",
            "189   f_item_view_1h_gap_code5   8",
            "190   f_item_view_3d_gap_code5   8    f_item_view_1h_gap_code5",
            "191   f_item_view_1w_gap_code5   8    f_item_view_1h_gap_code5",
            "192   f_type_view_1h_gap_code5   8    f_item_view_1h_gap_code5",
            "193   f_type_view_3d_gap_code5   8    f_item_view_1h_gap_code5",
            "194   f_cate_view_1h_gap_code8   8",
            "195   f_cate_view_1d_gap_code8   8    f_cate_view_1h_gap_code8",
            "196   f_cate_view_3d_gap_code8   8    f_cate_view_1h_gap_code8",
            "197   f_class_view_1d_gap_code8  8    f_cate_view_1h_gap_code8",
            "218   f_class_view_3d_gap_code8  8    f_cate_view_1h_gap_code8",
            "250   f_user_displayed_item_num  6",
            "252   f_user_dislayed_and_view_item_num   6",
            "253   f_user_displayed_and_order_item_num 6",
            "256   f_user_displayed_class_id  8",
            "257   f_user_displayed_cate_id   8",
            "258   f_user_displayed_and_view_class_id  8",
            "260   f_user_displayed_and_view_cate_id   8",
            "262   f_strategy  8",
            "263   f_req_page_num 6",
            '571  f_item_id_order_gap  4',  # 当前item_id距离上一次下单的gap
            '572  f_item_type_order_gap  4    f_item_id_order_gap',  # 当前item_type距离上一次下单的gap
            '573  f_item_cate_order_gap  4    f_item_id_order_gap',  # 当前item_cate距离上一次下单的gap
            '581  f_item_id_view_gap  4    f_item_id_order_gap',  # 当前item_id距离上一次点击的gap
            '582  f_item_type_view_gap  4    f_item_id_order_gap',  # 当前item_type距离上一次点击的gap
            '583  f_item_cate_view_gap  4    f_item_id_order_gap',  # 当前item_cate距离上一次点击的gap
            '522  f_item_id_view_scene_time_cnt  4',
            '526  f_item_type_view_scene_time_cnt  4    f_item_id_view_scene_time_cnt',
            '530  f_item_cate_view_scene_time_cnt  4    f_item_id_view_scene_time_cnt',
            '542  f_item_id_order_scene_time_cnt  4    f_item_id_view_scene_time_cnt',
            '546  f_item_type_order_scene_time_cnt  4    f_item_id_view_scene_time_cnt',
            '550  f_item_cate_order_scene_time_cnt  4    f_item_id_view_scene_time_cnt',
            '591  f_user_item_id_order_cnt  4',  # 购买订单数
            '592  f_user_item_id_redo_order_cnt  4',  # 复购订单数
            '593  f_user_item_id_order_kind  4',  # 该用户购买多少种商品
            '594  f_item_id_order_action_gap  4',  # 最近一次购买距今间隔（订单数量）
            '595  f_item_id_first_order_gap  4    f_item_id_order_gap',  # 第一次购买距今间隔（秒数）
            '596  f_item_id_first_order_action_gap  4    f_item_id_order_action_gap',  # 第一次购买距今间隔（订单数量）
            '597  f_item_id_last_v2_order_action_gap  4    f_item_id_order_action_gap',  # 最近一次购买商品 - 最后一次购买该商品（订单数量）
            '598  f_item_id_last_n_order_cnt  4    f_user_item_id_order_cnt',  # 最近N单购买次数
            '599  f_item_id_order_cnt  4    f_user_item_id_order_cnt',  # 该商品购买次数
            '127    f_scene_daytype 2',
            '128    f_scene_daynames 4',
            '129    f_weather_iconid 4',
            '130    f_weather_temp 4',
            '131    f_weather_temphigh 4',
            '132    f_weather_templow 4',
            '133    f_weather_pm25 4',
            '134    f_weather_zswd 4',
            '135    f_weather_windlevel   4',
            '136    f_weather_temprange 4',
            '137    f_loc2home_dishaversineindex 4',
            '138    f_loc2home_dismanhattanindex 4',
            '139    f_loc2home_diseuclideanindex 4',
            '140    f_loc2home_bearingvalue1index 4',
            '141    f_loc2home_bearingvalue2index 4',
            '142    f_loc2work_dishaversineindex 4',
            '143    f_loc2work_dismanhattanindex 4',
            '144    f_loc2work_diseuclideanindex 4',
            '145    f_loc2work_bearingvalue1index 4',
            '146    f_loc2work_bearingvalue2index 4',
            '147    f_loc3p_dissum 4',
            '148    f_loc3p_disarea 4',
            '149    f_dayofmonth 4',
            '150    f_dayofmonthrange4 4',
            '151    f_minuterange10 4',
            '351    f_ck_hr 4',
            '352    f_ck_hriid 4',
            '353    f_ck_hrdtype 4',
            '354    f_ck_hrclass 4',
            '355    f_ck_hrcate 4',
            '356    f_ck_hrtype 4',
            '357    f_ck_hriidrate 4',
            '358    f_ck_hrdtyperate 4',
            '359    f_ck_hrclassrate 4',
            '360    f_ck_hrcaterate 4',
            '361    f_ck_hrtyperate 4',
            '362    f_od_hr 4',
            '363    f_od_hriid 4',
            '364    f_od_hrdtype 4',
            '365    f_od_hrclass 4',
            '366    f_od_hrcate 4',
            '367    f_od_hrtype 4',
            '368    f_od_hriidrate 4',
            '369    f_od_hrdtyperate 4',
            '370    f_od_hrclassrate 4',
            '371    f_od_hrcaterate 4',
            '372    f_od_hrtyperate 4',
            '373    f_ck_last10_iid 4',
            '374    f_ck_last10_dtype 4',
            '375    f_ck_last10_classid 4',
            '376    f_ck_last10_cateid 4',
            '377    f_ck_last10_typeid 4',
            '378    f_ck_last20_iid 4',
            '379    f_ck_last20_dtype 4',
            '380    f_ck_last20_classid 4',
            '381    f_ck_last20_cateid 4',
            '382    f_ck_last20_typeid 4',
            '383    f_ck_last30_iid 4',
            '384    f_ck_last30_dtype 4',
            '385    f_ck_last30_classid 4',
            '386    f_ck_last30_cateid 4',
            '387    f_ck_last30_typeid 4',
            '992    f_caixiday_ck_hr 4',
            '993    f_caixiday_ck_hriid 4',
            '994    f_caixiday_ck_hrdtype 4',
            '995    f_caixiday_ck_hrclass 4',
            '996    f_caixiday_ck_hrcate 4',
            '997    f_caixiday_ck_hrtype 4',
            '998    f_caixiday_ck_hriidrate 4',
            '999    f_caixiday_ck_hrdtyperate 4',
            '1000    f_caixiday_ck_hrclassrate 4',
            '1001    f_caixiday_ck_hrcaterate 4',
            '1002    f_caixiday_ck_hrtyperate 4',
            '1003    f_caixiday_od_hr 4',
            '1004    f_caixiday_od_hriid 4',
            '1005    f_caixiday_od_hrdtype 4',
            '1006    f_caixiday_od_hrclass 4',
            '1007    f_caixiday_od_hrcate 4',
            '1008    f_caixiday_od_hrtype 4',
            '1009    f_caixiday_od_hriidrate 4',
            '1010    f_caixiday_od_hrdtyperate 4',
            '1011    f_caixiday_od_hrclassrate 4',
            '1012    f_caixiday_od_hrcaterate 4',
            '1013    f_caixiday_od_hrtyperate 4',
        ],
        "life_sta_list": [
            # "717 f_dtypeIdStr_click_cnt 4",
            # "718 f_dtypeiid_click_cycle_max 4",
            # "719 f_dtypeiid_click_cycle_min 4",
            # "720 f_dtypeiid_click_cycle_avg 4",
            # "721 f_dtype_click_cnt 4",
            # "753 f_dtype_click_cycle_max 4",
            # "723 f_dtype_click_cycle_min 4",
            # "724 f_dtype_click_cycle_avg 4",
            # # "725 f_bu_click_cnt 4",
            # # "726 f_bu_click_cycle_max 4",
            # # "727 f_bu_click_cycle_min 4",
            # # "728 f_bu_click_cycle_avg 4",
            # "729 f_firstid_click_cnt 4",
            # "730 f_firstid_click_cycle_max 4",
            # "731 f_firstid_click_cycle_min 4",
            # "732 f_firstid_click_cycle_avg 4",
            # "733 f_secondid_click_cnt 4",
            # "734 f_secondid_click_cycle_max 4",
            # "735 f_secondid_click_cycle_min 4",
            # "736 f_secondid_click_cycle_avg 4",
            # "737 f_thirdid_click_cnt 4",
            # "738 f_thirdid_click_cycle_max 4",
            # "739 f_thirdid_click_cycle_min 4",
            # "740 f_thirdid_click_cycle_avg 4",
            "786 f_lc_dtypeiid_cnt 4",
            "787 f_lc_dtypeiid_cycle_max 4",
            "788 f_lc_dtypeiid_cycle_min 4",
            "757 f_lc_dtypeiid_cycle_avg 4",
            "758 f_lc_dtype_cnt 4",
            "759 f_lc_dtype_cycle_max 4",
            "760 f_lc_dtype_cycle_min 4",
            "761 f_lc_dtype_cycle_avg 4",
            "762 f_lc_first_cnt 4",
            "763 f_lc_first_cycle_max 4",
            "764 f_lc_first_cycle_min 4",
            "765 f_lc_first_cycle_avg 4",
            "766 f_lc_second_cnt 4",
            "767 f_lc_second_cycle_max 4",
            "768 f_lc_second_cycle_min 4",
            "769 f_lc_second_cycle_avg 4",
            "770 f_lc_third_cnt 4",
            "771 f_lc_third_cycle_max 4",
            "772 f_lc_third_cycle_min 4",
            "773 f_lc_third_cycle_avg 4",
            "775 f_lc_dtypeiid_cnt_ratio 4",
            "776 f_lc_dtype_cnt_ratio 4",
            "779 f_lc_first_cnt_ratio 4",
            "780 f_lc_second_cnt_ratio 4",
            "785 f_lc_third_cnt_ratio 4",
            "797 lc_dtypeiid_30_days_click_cnt 4",
            "798 lc_dtypeiid_30_days_cnt_ratio 4",
            "799 lc_dtypeiid_14_days_click_cnt 4",
            "800 lc_dtypeiid_14_days_cnt_ratio 4",
            "801 lc_dtypeiid_7_days_click_cnt 4",
            "802 lc_dtypeiid_7_days_cnt_ratio 4",
            "803 lc_dtype_30_days_click_cnt 4",
            "804 lc_dtype_30_days_cnt_ratio 4",
            "805 lc_dtype_14_days_click_cnt 4",
            "806 lc_dtype_14_days_cnt_ratio 4",
            "807 lc_dtype_7_days_click_cnt 4",
            "808 lc_dtype_7_days_cnt_ratio 4",
            "809 lc_first_30_days_click_cnt 4",
            "810 lc_first_30_days_cnt_ratio 4",
            "811 lc_first_14_days_click_cnt 4",
            "812 lc_first_14_days_cnt_ratio 4",
            "813 lc_first_7_days_click_cnt 4",
            "814 lc_first_7_days_cnt_ratio 4",
            "815 lc_second_30_days_click_cnt 4",
            "816 lc_second_30_days_cnt_ratio 4",
            "817 lc_second_14_days_click_cnt 4",
            "818 lc_second_14_days_cnt_ratio 4",
            "819 lc_second_7_days_click_cnt 4",
            "820 lc_second_7_days_cnt_ratio 4",
            "821 lc_third_30_days_click_cnt 4",
            "822 lc_third_30_days_cnt_ratio 4",
            "823 lc_third_14_days_click_cnt 4",
            "824 lc_third_14_days_cnt_ratio 4",
            "825 lc_third_7_days_click_cnt 4",
            "826 lc_third_7_days_cnt_ratio 4",
            "827 geohash_click_cnt 4",
            "828 geohashDtype_click_cnt 4",
            "829 geohash_dtype_click_cycle_max 4",
            "830 geohash_dtype_click_cycle_min 4",
            "831 geohash_dtype_click_cycle_avg 4",
            "832 geohash_firstid_click_cnt 4",
            "833 geohash_first_click_cycle_max 4",
            "834 geohash_first_click_cycle_min 4",
            "835 geohash_first_click_cycle_avg 4",
            "836 geohash_secondid_click_cnt 4",
            "837 geohash_second_click_cycle_max 4",
            "838 geohash_second_click_cycle_min 4",
            "839 geohash_second_click_cycle_avg 4",
            "907 geohashItem_click_cnt 4",
            "908 geohash_itemid_click_cycle_max 4",
            "909 geohash_itemid_click_cycle_min 4",
            "910 geohash_itemid_click_cycle_avg 4",
        ],
        "life_item_list": [
            # "774 f_lc_ls_dt_3id_item 12"
        ],
        "life_seq_list": [
            "777 f_lc_ls_dt_iid  12 f_item_type_id",
            # "778 f_lc_ls_dt_3id 12 f_lc_ls_dt_3id_item",
            "781 f_lc_ls_tsgap 8",
            # "782 f_lc_ls_hour 8",
            # "783 f_lc_ls_week 8",
            # "784 f_lc_ls_geo7 8"
        ],
        "scene_sparse_list": [
        ],
        "waimai_sparse_list": [
        ],
        "item_sparse_list": [
            "403   f_mixlowordingfeaturev3   32",
            "404   f_mixhighordingfeaturev3  16",
            "722   f_sub_buz_id  4",
            "441    f_click_dtypepass_1year   8",
            "442    f_click_dtypeiidpass_1year   8",
            "445    f_click_typeidpass_1year   8",
            "447    f_order_dtypepass_1year   8",
            "448    f_order_dtypeiidpass_1year   8",
            "451    f_order_typeidpass_1year   8",
        ],
        "i2i_sparse_list": [
            "482    f_i2i_sparse    12    f_item_type_id"
        ],
        # seq特征
        "seq_max_len": 20,  # 这批特征序列长度最大是20，且按时间倒序。
        "seq_emb_len": 20,
        "click_seq_key_list": [
            "454    f_click_freq_seq_dtype_20   20",
            "468    f_click_dtypeseq_20   20    f_click_freq_seq_dtype_20",
            # new
            "912    f_guessclick_freq_seq_dtype_20   20 f_click_freq_seq_dtype_20",
            "942    f_guessclick_dtypeseq_20   20   f_click_freq_seq_dtype_20",
        ],
        "click_seq_query_list": [
            "456    f_click_freq_seq_week_20  20",
            "457    f_click_freq_seq_hour_20  20",
            "458    f_click_freq_seq_timegap_20   20",
            "471    f_click_weekseq_20  20    f_click_freq_seq_week_20",
            "470    f_click_hourseq_20  20    f_click_freq_seq_hour_20",
            "472    f_click_gapseq_20   20    f_click_freq_seq_timegap_20",
            # new
            "914    f_guessclick_freq_seq_week_20  20   f_click_freq_seq_week_20",
            "915    f_guessclick_freq_seq_hour_20  20   f_click_freq_seq_hour_20",
            "916    f_guessclick_freq_seq_timegap_20   20   f_click_freq_seq_timegap_20",
            "944    f_guessclick_weekseq_20  20 f_click_freq_seq_week_20",
            "945    f_guessclick_hourseq_20  20 f_click_freq_seq_hour_20",
            "946    f_guessclick_gapseq_20   20 f_click_freq_seq_timegap_20",
        ],
        "order_seq_key_list": [
            "461    f_order_freq_seq_dtype_20   20  f_click_freq_seq_dtype_20",
            "475    f_order_dtypeseq_20   20    f_click_freq_seq_dtype_20",
            # new
            "919    f_guessorder_freq_seq_dtype_20   20 f_click_freq_seq_dtype_20",
            "949    f_guessorder_dtypeseq_20   20   f_click_freq_seq_dtype_20",
        ],
        "order_seq_query_list": [
            "463    f_order_freq_seq_week_20    20  f_click_freq_seq_week_20",
            "464    f_order_freq_seq_hour_20    20  f_click_freq_seq_hour_20",
            "465    f_order_freq_seq_timegap_20   20    f_click_freq_seq_timegap_20",
            "478    f_order_weekseq_20    20    f_click_freq_seq_week_20",
            "477    f_order_hourseq_20    20    f_click_freq_seq_hour_20",
            "479    f_order_gapseq_20   20    f_click_freq_seq_timegap_20",
            # new
            "921    f_guessorder_freq_seq_week_20  20   f_click_freq_seq_week_20",
            "922    f_guessorder_freq_seq_hour_20  20   f_click_freq_seq_hour_20",
            "923    f_guessorder_freq_seq_timegap_20   20   f_click_freq_seq_timegap_20",
            "951    f_guessorder_weekseq_20  20 f_click_freq_seq_week_20",
            "952    f_guessorder_hourseq_20  20 f_click_freq_seq_hour_20",
            "953    f_guessorder_gapseq_20   20 f_click_freq_seq_timegap_20",
        ]
    }
}
# 用户序列
click_seq_key_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['click_seq_key_list']]
click_seq_query_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['click_seq_query_list']]
order_seq_key_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['order_seq_key_list']]
order_seq_query_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['order_seq_query_list']]
click_seq_masks_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['click_seq_key_list']]
order_seq_masks_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['order_seq_key_list']]
# i2i序列
i2i_seq_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['i2i_sparse_list']]
# 用户终身序列
life_seq_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['life_seq_list']]
life_sta_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['life_sta_list']]
life_item_namesstr = [str(lines.split()[1]) for lines in conf['sparse']['life_item_list']]

########## feature groups ####################
common_features = [
    'f_gender', 'f_age', 'f_constellation', 'f_married', 'f_job',
    'f_user_level', 'f_cityid', 'f_loc_cityid', 'f_client_type',
    'f_reslution', 'f_agent', 'f_app_source', 'f_app_version', 'f_user_id', 'f_userid0',
    'f_item_type_id', 'f_dtype',
    'f_geohash5', 'f_geohash',
    'f_isremote', 'f_remote_type', 'f_user_view_item_dtype_lists_6h',
    'f_user_view_class_lists_6h', 'f_user_view_type_lists_6h', 'f_user_view_cate_lists_6h',
    'f_user_order_brandid_lists_6h',
    'f_user_view_brandid_lists_6h', 'f_user_view_item_dtype_lists_24h', 'f_user_view_item_dtype_lists',
    'f_user_order_class_lists_24h',
    'f_user_order_type_lists_24h', 'f_user_view_class_lists_24h', 'f_user_view_type_lists_24h',
    'f_user_view_cate_lists_24h', 'f_user_order_brandid_lists_24h', 'f_user_view_brandid_lists_24h',
    'f_user_order_item_dtype_lists', 'f_user_order_class_lists',
    'f_user_order_type_lists', 'f_user_order_cate_lists', 'f_user_view_class_lists',
    'f_user_view_type_lists', 'f_user_view_cate_lists', 'f_user_order_brandid_lists',
    'f_user_view_brandid_lists', 'f_strategy', 'f_req_page_num',
    'f_click_dtypepass_1year', 'f_click_dtypeiidpass_1year',
    'f_click_typeidpass_1year',
    'f_order_dtypepass_1year', 'f_order_dtypeiidpass_1year',
    'f_order_typeidpass_1year',
    'f_item_id_order_gap',  # 当前item_id距离上一次下单的gap
    'f_item_type_order_gap',  # 当前item_type距离上一次下单的gap
    'f_item_cate_order_gap',  # 当前item_cate距离上一次下单的gap
    'f_item_id_view_gap',  # 当前item_id距离上一次点击的gap
    'f_item_type_view_gap',  # 当前item_type距离上一次点击的gap
    'f_item_cate_view_gap',  # 当前item_cate距离上一次点击的gap
    'f_item_id_view_scene_time_cnt',
    'f_item_type_view_scene_time_cnt',
    'f_item_cate_view_scene_time_cnt',
    'f_item_id_order_scene_time_cnt',
    'f_item_type_order_scene_time_cnt',
    'f_item_cate_order_scene_time_cnt',
    'f_user_item_id_order_cnt',  # 购买订单数
    'f_user_item_id_redo_order_cnt',  # 复购订单数
    'f_user_item_id_order_kind',  # 该用户购买多少种商品
    'f_item_id_order_action_gap',  # 最近一次购买距今间隔（订单数量）
    'f_item_id_first_order_gap',  # 第一次购买距今间隔（秒数）
    'f_item_id_first_order_action_gap',  # 第一次购买距今间隔（订单数量）
    'f_item_id_last_v2_order_action_gap',  # 最近一次购买商品 - 最后一次购买该商品（订单数量）
    'f_item_id_last_n_order_cnt',  # 最近N单购买次数
    'f_item_id_order_cnt',  # 该商品购买次数
    'f_ck_hr',
    'f_ck_hriid',
    'f_ck_hrdtype',
    'f_ck_hrclass',
    'f_ck_hrcate',
    'f_ck_hrtype',
    'f_ck_hriidrate',
    'f_ck_hrdtyperate',
    'f_ck_hrclassrate',
    'f_ck_hrcaterate',
    'f_ck_hrtyperate',
    'f_od_hr',
    'f_od_hriid',
    'f_od_hrdtype',
    'f_od_hrclass',
    'f_od_hrcate',
    'f_od_hrtype',
    'f_od_hriidrate',
    'f_od_hrdtyperate',
    'f_od_hrclassrate',
    'f_od_hrcaterate',
    'f_od_hrtyperate',
    'f_ck_last10_iid',
    'f_ck_last10_dtype',
    'f_ck_last10_classid',
    'f_ck_last10_cateid',
    'f_ck_last10_typeid',
    'f_ck_last20_iid',
    'f_ck_last20_dtype',
    'f_ck_last20_classid',
    'f_ck_last20_cateid',
    'f_ck_last20_typeid',
    'f_ck_last30_iid',
    'f_ck_last30_dtype',
    'f_ck_last30_classid',
    'f_ck_last30_cateid',
    'f_ck_last30_typeid',
    'f_caixiday_ck_hr',
    'f_caixiday_ck_hriid',
    'f_caixiday_ck_hrdtype',
    'f_caixiday_ck_hrclass',
    'f_caixiday_ck_hrcate',
    'f_caixiday_ck_hrtype',
    'f_caixiday_ck_hriidrate',
    'f_caixiday_ck_hrdtyperate',
    'f_caixiday_ck_hrclassrate',
    'f_caixiday_ck_hrcaterate',
    'f_caixiday_ck_hrtyperate',
    'f_caixiday_od_hr',
    'f_caixiday_od_hriid',
    'f_caixiday_od_hrdtype',
    'f_caixiday_od_hrclass',
    'f_caixiday_od_hrcate',
    'f_caixiday_od_hrtype',
    'f_caixiday_od_hriidrate',
    'f_caixiday_od_hrdtyperate',
    'f_caixiday_od_hrclassrate',
    'f_caixiday_od_hrcaterate',
    'f_caixiday_od_hrtyperate',
]
sz_session_0_features = [
    'f_user_1h_most_view_cate', 'f_user_1h_most_view_type',
    'f_user_1h_most_view_class', 'f_user_1h_most_two_view_cate',
    'f_user_1h_most_three_view_class', 'f_user_6h_most_view_type',
    'f_user_6h_most_three_view_class_list', 'f_user_1w_most_three_view_class'
]
sz_session_1_features = [
    'f_1d_last_view_two_cates', 'f_1d_last_view_three_class',
    'f_3d_last_view_cates', 'f_3d_last_view_two_class',
    'f_3d_last_view_three_class', 'f_1w_last_view_cates',
    'f_1w_last_view_two_class', 'f_all_last_view_cates'
]
sz_session_3_features = [
    'f_2h_last_order_types', 'f_2h_last_order_class',
    'f_2h_last_order_two_cates', 'f_12h_last_order_two_types',
    'f_2d_last_order_three_class'
]
sz_session_5_features = [
    'f_item_view_1h_gap_code5', 'f_item_view_3d_gap_code5',
    'f_item_view_1w_gap_code5', 'f_type_view_1h_gap_code5',
    'f_type_view_3d_gap_code5', 'f_cate_view_1h_gap_code8',
    'f_cate_view_1d_gap_code8', 'f_cate_view_3d_gap_code8',
    'f_class_view_1d_gap_code8', 'f_class_view_3d_gap_code8'
]
user_displayed_item_num_features = [
    'f_user_displayed_item_num', 'f_user_dislayed_and_view_item_num', 'f_user_displayed_and_order_item_num'
]
i2i_features = [
    'f_i2i_sparse'
]
user_displayed_item_id_features = [
    'f_user_displayed_class_id', 'f_user_displayed_cate_id',
    'f_user_displayed_and_view_class_id', 'f_user_displayed_and_view_cate_id'
]
mix_features = [
    'f_mixlowordingfeaturev3', 'f_mixhighordingfeaturev3', 'f_sub_buz_id'
]
scene_features = [
    'f_isremote', 'f_remote_type', 'f_loc_cityid',
    'f_scene_daytype',
    'f_scene_daynames',
    'f_weather_iconid',
    'f_weather_temp',
    'f_weather_temphigh',
    'f_weather_templow',
    'f_weather_pm25',
    'f_weather_zswd',
    'f_weather_windlevel',
    'f_weather_temprange',
    'f_loc2home_dishaversineindex',
    'f_loc2home_dismanhattanindex',
    'f_loc2home_diseuclideanindex',
    'f_loc2home_bearingvalue1index',
    'f_loc2home_bearingvalue2index',
    'f_loc2work_dishaversineindex',
    'f_loc2work_dismanhattanindex',
    'f_loc2work_diseuclideanindex',
    'f_loc2work_bearingvalue1index',
    'f_loc2work_bearingvalue2index',
    'f_loc3p_dissum',
    'f_loc3p_disarea',
    'f_dayofmonth',
    'f_dayofmonthrange4',
    'f_minuterange10',
]
item_info_features = ['f_item_type_id', 'f_dtype', 'f_sub_buz_id']

"""
  graph  state helper 
"""
extra_graph_states = [
    graph_state_helper('TRAIN'),
    graph_state_helper('EVALUATE'),
    graph_state_helper('PREDICT', 'default'),
    graph_state_helper('PREDICT', 'item_precompute')
]
########## feature groups ###################
def build_model():
    # handle dense feature
    dense_len = int(conf['dense']['dense_len'])
    # sparse feature column定义和linear部分
    user_sparse_list = conf['sparse']['user_sparse_list']
    scene_sparse_list = conf['sparse']['scene_sparse_list']
    item_sparse_list = conf['sparse']['item_sparse_list']
    waimai_sparse_list = conf['sparse']['waimai_sparse_list']
    user_seq_list = conf['sparse']['click_seq_key_list'] + \
                    conf['sparse']['click_seq_query_list'] + \
                    conf['sparse']['order_seq_key_list'] + \
                    conf['sparse']['order_seq_query_list'] + \
                    conf['sparse']['i2i_sparse_list']

    life_seq_1000_list = conf['sparse']['life_seq_list']
    global g_mlx_feature_names
    global g_mlx_embed_names
    global g_mlx_embed_mapping
    global g_mlx_seq_feature_names
    global g_mlx_embed_seq_mapping
    global g_mlx_embed_mask_mapping
    g_mlx_feature_names = []
    g_mlx_embed_names = []
    g_mlx_embed_mapping = {}
    g_mlx_seq_feature_names = []
    g_mlx_embed_seq_mapping = {}
    g_mlx_embed_mask_mapping = {}
    sparse_features = []

    life_sta_list = conf['sparse']['life_sta_list']
    life_item_list = conf['sparse']['life_item_list']

    # 1:通过colid。对所有的sparse特征构建tensor，并且加入到linear部分。
    for i, fea_info in enumerate(
            user_sparse_list + scene_sparse_list + item_sparse_list + waimai_sparse_list + \
            life_sta_list + life_item_list
    ):
        cid = int(fea_info.split()[0])
        name = str(fea_info.split()[1])
        _ = int(fea_info.split()[2])
        sparse_feature = tfmlx.create_sparse_feature(cid, name)
        sparse_features.append(sparse_feature)
    # 2:通过colid。想单独处理seq特征。seq没有办法不加入到线性部分，mlx并不支持。调用tfmlx.create_sparse_feature一定会构建线性部分！
    for i, fea_info in enumerate(user_seq_list + life_seq_1000_list):
        cid = int(fea_info.split()[0])
        name = str(fea_info.split()[1])
        _ = int(fea_info.split()[2])
        sparse_feature = tfmlx.create_sparse_feature(cid, name)
        sparse_features.append(sparse_feature)
    # 3:把线性部分的tensor的col信息提取出来，后面会转tensor。
    linear_column = tfmlx.linear_column(sparse_features)
    # 4:把所有的sparse特征压入map，namestr->tensor
    for i, fea_info in enumerate(user_sparse_list + life_sta_list + life_item_list):
        items = fea_info.split()
        name = str(items[1])
        dim = int(items[2])
        share_embed_with = None
        if len(items) > 3:  # 共享embedding
            share_embed_name = items[3]
            share_embed_with = g_mlx_embed_mapping[share_embed_name].name
        if dim > 0:
            if share_embed_with is not None:
                print('share_emb', name, share_embed_name, dim)
            f = tfmlx.get_sparse_feature_by_name(name)
            emb_var = tfmlx.embedding_column(f, dimensions=dim, combiner='sum',
                                             share_embed_with=share_embed_with)
            emb_output = emb_var.get_output()
            g_mlx_feature_names.append(name)
            g_mlx_embed_names.append(emb_output.name)
            g_mlx_embed_mapping[name] = emb_output
            g_mlx_embed_mapping[emb_output.name] = emb_output
    for i, fea_info in enumerate(scene_sparse_list):
        _ = int(fea_info.split()[0])
        name = str(fea_info.split()[1])
        dim = int(fea_info.split()[2])
        if dim > 0:
            sf = tfmlx.get_sparse_feature_by_name(name)
            emb_var = tfmlx.embedding_column(sf, dimensions=dim, combiner='sum')
            emb_output = emb_var.get_output()
            g_mlx_feature_names.append(name)
            g_mlx_embed_names.append(emb_output.name)
            g_mlx_embed_mapping[name] = emb_output
            g_mlx_embed_mapping[emb_output.name] = emb_output
    for i, fea_info in enumerate(item_sparse_list):
        items = fea_info.split()
        name = str(items[1])
        dim = int(items[2])
        share_embed_with = None
        if len(items) > 3:  # 共享embedding
            share_embed_name = items[3]
            share_embed_with = g_mlx_embed_mapping[share_embed_name].name
        if dim > 0:
            if share_embed_with is not None:
                print('share_emb', name, share_embed_name, dim)
            f = tfmlx.get_sparse_feature_by_name(name)
            emb_var = tfmlx.embedding_column(f, dimensions=dim, combiner='sum',
                                             share_embed_with=share_embed_with)
            emb_output = emb_var.get_output()
            g_mlx_feature_names.append(name)
            g_mlx_embed_names.append(emb_output.name)
            g_mlx_embed_mapping[name] = emb_output
            g_mlx_embed_mapping[emb_output.name] = emb_output
    for i, fea_info in enumerate(waimai_sparse_list):
        _ = int(fea_info.split()[0])
        name = str(fea_info.split()[1])
        dim = int(fea_info.split()[2])
        if dim > 0:
            sf = tfmlx.get_sparse_feature_by_name(name)
            emb_var = tfmlx.embedding_column(sf, dimensions=dim, combiner='sum')
            emb_output = emb_var.get_output()
            g_mlx_feature_names.append(name)
            g_mlx_embed_names.append(emb_output.name)
            g_mlx_embed_mapping[name] = emb_output
            g_mlx_embed_mapping[emb_output.name] = emb_output
    """
     序列特征初始化
    """
    # 5:序列特征压入list和map  g_mlx_seq_feature_names  g_mlx_embed_seq_mapping  g_mlx_embed_mask_mapping
    seq_max_len = conf['sparse']['seq_max_len']
    for i, fea_info in enumerate(user_seq_list):
        items = fea_info.split()
        name = str(items[1])
        dim = int(items[2])
        share_embed_with = None
        if len(items) > 3:  # 共享embedding
            share_embed_name = items[3]
            share_embed_with = g_mlx_embed_mapping[share_embed_name].name
            if share_embed_with is not None:
                print('share_emb', name, share_embed_name, dim)
        f = tfmlx.get_sparse_feature_by_name(name)
        max_seq_len = seq_max_len
        if 'lifeseq' in name:
            max_seq_len = 100
        emb_var = tfmlx.embedding_column(f, dimensions=dim, combiner='concat_1d', max_seq_len=max_seq_len,
                                         share_embed_with=share_embed_with)
        emb_output = emb_var.get_output()
        emb_mask = emb_var.get_seq_mask()
        g_mlx_seq_feature_names.append(name)
        g_mlx_embed_seq_mapping[name] = emb_output
        g_mlx_embed_mask_mapping[name] = emb_mask
        g_mlx_embed_mapping[name] = emb_output
        g_mlx_embed_mapping[emb_output.name] = emb_output

    seq_1000_len = 100
    for i, fea_info in enumerate(life_seq_1000_list):
        items = fea_info.split()
        name = str(items[1])
        dim = int(items[2])
        share_embed_with = None
        if len(items) > 3:  # 共享embedding
            share_embed_name = items[3]
            share_embed_with = g_mlx_embed_mapping[share_embed_name].name
            if share_embed_with is not None:
                print('share_emb', name, share_embed_name, dim)
        f = tfmlx.get_sparse_feature_by_name(name)
        emb_var = tfmlx.embedding_column(f, dimensions=dim, combiner='concat_1d', max_seq_len=seq_1000_len,
                                         share_embed_with=share_embed_with)
        emb_output = emb_var.get_output()
        emb_mask = emb_var.get_seq_mask()
        g_mlx_seq_feature_names.append(name)
        g_mlx_embed_seq_mapping[name] = emb_output
        g_mlx_embed_mask_mapping[name] = emb_mask
        g_mlx_embed_mapping[name] = emb_output
        g_mlx_embed_mapping[emb_output.name] = emb_output



    # 6:把dense的tensor提取出来
    dense_input = tfmlx.get_dense(cid=1024, dense_len=dense_len)
    # 7:把星期和小时做了交叉，放入线性部分。没什么用啊，线性部分都是0/1，存在不存在，哪里保证加入到common的embeding里了？？？
    def add_cross_features(l_features, r_features, next_id):
        for name_l in l_features:
            for name_r in r_features:
                f_l = tfmlx.get_sparse_feature_by_name(name_l)
                f_r = tfmlx.get_sparse_feature_by_name(name_r)
                feature = tfmlx.create_crossed_sparse_feature(next_id, f_l, f_r)
                linear_column.add_sparse_feature(feature)
                next_id -= 1
        return next_id
    next_id = 2000
    next_id = add_cross_features(['f_weekday'], ['f_hour'], next_id)
    linear_output = linear_column.get_output()  # 线性部分的tensor拿出来了！
    # 8:把一些特征组合保存到map中！！！在模型中会通过组合拿出来！
    global g_my_name_collections
    g_my_name_collections = {}
    """
        seq特征传递 query特征传递---------------------------------------------------------------------#
    """
    g_my_name_collections['click_seq_key_embs_put'] = click_seq_key_namesstr
    g_my_name_collections['click_seq_query_embs_put'] = click_seq_query_namesstr
    g_my_name_collections['order_seq_key_embs_put'] = order_seq_key_namesstr
    g_my_name_collections['order_seq_query_embs_put'] = order_seq_query_namesstr
    g_my_name_collections['click_seq_masks_put'] = click_seq_masks_namesstr
    g_my_name_collections['order_seq_masks_put'] = order_seq_masks_namesstr
    # ------------------------------------------------------------------------------------------------#
    g_my_name_collections['i2i_seq_embeds'] = i2i_seq_namesstr

    g_my_name_collections['life_seq_embeds'] = life_seq_namesstr
    g_my_name_collections['life_sta_embeds'] = life_sta_namesstr
    g_my_name_collections['life_item_embeds'] = life_item_namesstr

    g_my_name_collections['common_embeds'] = common_features
    g_my_name_collections['sz_session_0_embeds'] = sz_session_0_features
    g_my_name_collections['sz_session_1_embeds'] = sz_session_1_features
    g_my_name_collections['sz_session_3_embeds'] = sz_session_3_features
    g_my_name_collections['sz_session_5_embeds'] = sz_session_5_features
    g_my_name_collections['user_displayed_item_num_embeds'] = user_displayed_item_num_features
    g_my_name_collections['user_displayed_item_id_embeds'] = user_displayed_item_id_features
    g_my_name_collections['mix_feature_embeds'] = mix_features
    g_my_name_collections['scene_embeds'] = scene_features
    g_my_name_collections['item_info_embeds'] = item_info_features
    # 标签
    label_ctr = tfmlx.get_labels(indexes=[0], name="label_0")
    label_ctcvr = tfmlx.get_labels(indexes=[1], name="label_1")
    # 样本权重
    sample_weight_ctr = tfmlx.get_sample_weights(indexes=[0], name="sample_weight_0")
    sample_weight_ctcvr = tfmlx.get_sample_weights(indexes=[1], name="sample_weight_1")
    label_aux = tfmlx.get_labels(indexes=[2], name="label_2")
    sample_weight_aux = tfmlx.get_sample_weights(indexes=[2], name="sample_weight_2")
    is_training = True
    # model函数的重点是传递tensor！！！
    model(dense_input,
          linear_output,
          [g_mlx_embed_mapping[x] for x in g_mlx_feature_names],  # 必须保证顺序！保存的是tensor
          [g_mlx_embed_seq_mapping[x] for x in g_mlx_seq_feature_names],
          [g_mlx_embed_mask_mapping[x] for x in g_mlx_seq_feature_names],
          label_ctr,
          label_ctcvr,
          label_aux,
          sample_weight_ctr,
          sample_weight_ctcvr,
          sample_weight_aux,
          is_training)
    adam_opt = tfmlx.Adam(learning_rate=1e-5, l2_regularization=1e-7)
    ftrl_opt = tfmlx.Ftrl(learning_rate=1e-4, l1_regularization=1e-6)
    sadam_opt = tfmlx.Sadam(learning_rate=1e-5, l2_regularization=1e-7)
    ftrl_opt.optimize(tfmlx.LINEAR_VARIABLES)
    adam_opt.optimize(tfmlx.GRAPH_VARIABLES)
    sadam_opt.optimize(tfmlx.EMBEDDING_VARIABLES)
    tfmlx.set_filter(capacity=(1 << 27), min_cnt=5, cbf=True, reset_percent=99)
    tfmlx.set_col_max_train_epoch(1)
    return tfmlx.get_model()


@tfmlx.tf_wrapper(extra_graph_states, no_default_states = True)
def model(dense_input, linear, embeddings, seq_embeddings, seq_embeddings_mask, label_ctr, label_ctcvr, label_aux,
          sample_weight_ctr, sample_weight_ctcvr, sample_weight_aux, is_training=True, **kwargs):
    global g_mlx_feature_names
    global g_mlx_seq_feature_names
    namestr2tensor_embed_mapping = {
        x[0]: x[1] for x in zip(g_mlx_feature_names, embeddings)
    }
    namestr2tensor_seq_mapping = {
        x[0]: x[1] for x in zip(g_mlx_seq_feature_names, seq_embeddings)
    }
    namestr2tensor_seqmask_mapping = {
        x[0]: x[1] for x in zip(g_mlx_seq_feature_names, seq_embeddings_mask)
    }
    def get_my_collection(name):
        global g_my_name_collections
        if name not in g_my_name_collections:
            raise Exception('Cannot find collection: ' + name)
        return [namestr2tensor_embed_mapping[x] for x in
                g_my_name_collections[name]]  # g_my_name_collections的value是tensor.name
    def get_emb_seq_collection(name):
        global g_my_name_collections
        if name not in g_my_name_collections:
            raise Exception('Cannot find collection in get_seq_collection: ' + name)
        return [namestr2tensor_seq_mapping[x] for x in g_my_name_collections[name]]
    def get_mask_seq_collection(name):
        global g_my_name_collections
        if name not in g_my_name_collections:
            raise Exception('Cannot find collection in get_seq_collection: ' + name)
        return [namestr2tensor_seqmask_mapping[x] for x in g_my_name_collections[name]]
    # 0：对需要split的特征进行split
    hour_splits = tf.split(tf.nn.relu(add_fc(namestr2tensor_embed_mapping["f_hour"], 36, None, 'hour_fc')), 6, axis=-1)
    week_splits = tf.split(tf.nn.relu(add_fc(namestr2tensor_embed_mapping["f_weekday"], 36, None, 'week_fc')), 6,
                           axis=-1)
    for index, print_tensor in enumerate(hour_splits):
        print("打印切分的hour-" + str(index) + "-" + str(print_tensor))
    # 1:把最初定义的特征组合取出来，放到list中。
    common_embeds = get_my_collection('common_embeds')
    sz_session_0_embeds = get_my_collection('sz_session_0_embeds')
    sz_session_1_embeds = get_my_collection('sz_session_1_embeds')
    sz_session_3_embeds = get_my_collection('sz_session_3_embeds')
    sz_session_5_embeds = get_my_collection('sz_session_5_embeds')
    user_displayed_item_num_embeds = get_my_collection('user_displayed_item_num_embeds')
    user_displayed_item_id_embeds = get_my_collection('user_displayed_item_id_embeds')
    # 2：把sum-pooling的特征sum起来，生成tensor。都放到list-embeds_to_concat中
    def add_sum(name, inputs):
        return tf.add_n(inputs, name=name)
    sz_session_0_list_sum = add_sum('sz_session_0_list_sum', sz_session_0_embeds)
    sz_session_1_list_sum = add_sum('sz_session_1_list_sum', sz_session_1_embeds)
    sz_session_3_list_sum = add_sum('sz_session_3_list_sum', sz_session_3_embeds)
    sz_session_5_list_sum = add_sum('sz_session_5_list_sum', sz_session_5_embeds)
    user_displayed_item_nums = add_sum('user_displayed_item_nums', user_displayed_item_num_embeds)
    user_displayed_item_ids = add_sum('user_displayed_item_ids', user_displayed_item_id_embeds)
    # i2i network
    SEQ_MAX_LEN = 20
    def reshape_seq(seq, seq_max_len=SEQ_MAX_LEN):
        return [tf.reshape(x, [-1, seq_max_len, x.shape[-1] // seq_max_len]) for x in seq]
    i2i_seq_embeds_list = reshape_seq(get_emb_seq_collection('i2i_seq_embeds'))
    i2i_seq_mask = get_mask_seq_collection('i2i_seq_embeds')[0]
    if len(i2i_seq_embeds_list) > 1:
        i2i_seq_embeds = tf.concat(i2i_seq_embeds_list, axis=-1)
    else:
        i2i_seq_embeds = i2i_seq_embeds_list[0]
    i2i_querys_concat = tf.concat([hour_splits[0], week_splits[0]], axis=1)
    i2i_seq_emb_size = int(i2i_seq_embeds.shape[-1])
    print("i2i_seq_embeds",str(i2i_seq_embeds))
    i2i_attention_tensor = ContextualMultiHeadMulAttention2('i2i_attention', [64, i2i_seq_emb_size], 4,
                                                            activation='silu')(
        [i2i_seq_embeds,
         i2i_seq_embeds,
         i2i_querys_concat,
         i2i_seq_mask]
    )


    # life network
    ## life_sta, life_item
    life_sta_embeds = get_my_collection('life_sta_embeds')
    life_item_embeds = get_my_collection('life_item_embeds')
    print("life_item_embeds",str(life_item_embeds))
    # life_item_embed_to_query = tf.concat(life_item_embeds,axis = -1)

    ## life_seq
    life_seq_max_len = 100
    life_seq_embeds_list = reshape_seq(get_emb_seq_collection('life_seq_embeds'), life_seq_max_len)
    life_seq_mask = get_mask_seq_collection('life_seq_embeds')[0]
    life_seq_embeds = tf.concat(life_seq_embeds_list, axis=-1)

    print("life_seq_embeds",str(life_seq_embeds))
    life_seq_embeds =  tf.split(life_seq_embeds,[20,80],axis = 1)[0]
    print("life_seq_embeds",str(life_seq_embeds))

    print("life_seq_mask",str(life_seq_mask))
    life_seq_mask = tf.split(life_seq_mask,[20,80],axis = 1)[0]
    print("life_seq_mask",str(life_seq_mask))

    item_info_emb = get_my_collection('item_info_embeds')
    item_info_embs = tf.concat(life_item_embeds + item_info_emb,axis = -1)
    print("item_info_embs",str(item_info_embs))

    life_seq_emb_size = int(life_seq_embeds.shape[-1])
    life_attention_tensor = ContextualMultiHeadMulAttention2('life_attention', [64, life_seq_emb_size], 4,
                                                             activation='relu')(
        [life_seq_embeds,
         life_seq_embeds,
         item_info_embs,
         life_seq_mask]
    )

    life_out = life_sta_embeds + [life_attention_tensor]
    life_out_vector = tf.concat(life_out, axis=-1)

    """
           SEQ NETWORK
    """
    click_seq_masks_put = get_mask_seq_collection('click_seq_masks_put')
    order_seq_masks_put = get_mask_seq_collection('order_seq_masks_put')
    click_seq_mask_long = click_seq_masks_put[0]
    order_seq_mask_long = order_seq_masks_put[0]
    click_seq_mask_short = click_seq_masks_put[1]  # 要根据位置修改索引
    order_seq_mask_short = order_seq_masks_put[1]  # 要根据位置修改索引
    click_seq_mask_caixi = click_seq_masks_put[2]
    order_seq_mask_caixi = order_seq_masks_put[2]
    click_seq_mask_caixi_short = click_seq_masks_put[3]
    order_seq_mask_caixi_short = order_seq_masks_put[3]
    # 8.0: 序列part-two---------------------------------------------------------------------------------------------------------
    hour_week_concat1 = tf.concat([hour_splits[2], week_splits[2]], axis=1)
    hour_week_concat2 = tf.concat([hour_splits[3], week_splits[3]], axis=1)
    hour_week_concat3 = tf.concat([hour_splits[4], week_splits[4]], axis=1)
    hour_week_concat4 = tf.concat([hour_splits[5], week_splits[5]], axis=1)
    long_click_querys_concat = tf.tile(tf.expand_dims(hour_week_concat1, 1), [1, conf['sparse']['seq_emb_len'], 1])
    long_order_querys_concat = tf.tile(tf.expand_dims(hour_week_concat2, 1), [1, conf['sparse']['seq_emb_len'], 1])
    short_click_querys_concat = tf.tile(tf.expand_dims(hour_week_concat3, 1), [1, conf['sparse']['seq_emb_len'], 1])
    short_order_querys_concat = tf.tile(tf.expand_dims(hour_week_concat4, 1), [1, conf['sparse']['seq_emb_len'], 1])
    # ~------------- caixi long click/order seq process
    caixi_click_keys_seqs_infos = [
        'f_guessclick_freq_seq_week_20',
        'f_guessclick_freq_seq_hour_20',
        'f_guessclick_freq_seq_timegap_20'
    ]
    caixi_click_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                             [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                  for namestr in caixi_click_keys_seqs_infos]
    caixi_click_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_guessclick_freq_seq_dtype_20'],
                                              [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    caixi_order_keys_seqs_infos = [
        'f_guessorder_freq_seq_week_20',
        'f_guessorder_freq_seq_hour_20',
        'f_guessorder_freq_seq_timegap_20'
    ]
    caixi_order_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                             [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                  for namestr in caixi_order_keys_seqs_infos]
    caixi_order_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_guessorder_freq_seq_dtype_20'],
                                              [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    # --
    caixi_click_attention_tensor_long = ContextualMultiHeadMulAttention('caixi_click_attention', [64, 20], 4,
                                                                        activation='silu')(
        [caixi_click_values_seqs_embs,
         caixi_click_keys_seqs_embs,
         long_click_querys_concat,
         click_seq_mask_caixi]
    )
    print("caixi long点击的长度" + str(caixi_click_attention_tensor_long))
    caixi_order_attention_tensor_long = ContextualMultiHeadMulAttention('caixi_order_attention', [64, 20], 4,
                                                                        activation='silu')(
        [caixi_order_values_seqs_embs,
         caixi_order_keys_seqs_embs,
         long_order_querys_concat,
         order_seq_mask_caixi]
    )
    print("caixi long下单的长度：" + str(caixi_order_attention_tensor_long))
    # ~------------- caixi short click/order seq process
    caixi_short_click_keys_seqs_infos = [
        'f_guessclick_weekseq_20',
        'f_guessclick_hourseq_20',
        'f_guessclick_gapseq_20'
    ]
    caixi_short_click_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                                   [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                        for namestr in caixi_short_click_keys_seqs_infos]
    caixi_short_click_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_guessclick_dtypeseq_20'],
                                                    [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    caixi_short_order_keys_seqs_infos = [
        "f_guessorder_weekseq_20",
        "f_guessorder_hourseq_20",
        "f_guessorder_gapseq_20"
    ]
    caixi_short_order_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                                   [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                        for namestr in caixi_short_order_keys_seqs_infos]
    caixi_short_order_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_guessorder_dtypeseq_20'],
                                                    [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    # --
    caixi_short_click_attention_tensor_long = ContextualMultiHeadMulAttention('caixi_short_click_attention', [64, 20],
                                                                              4,
                                                                              activation='silu')(
        [caixi_short_click_values_seqs_embs,
         caixi_short_click_keys_seqs_embs,
         short_click_querys_concat,
         click_seq_mask_caixi_short]
    )
    print("caixi short点击的长度：" + str(caixi_short_click_attention_tensor_long))
    caixi_short_order_attention_tensor_long = ContextualMultiHeadMulAttention('caixi_short_order_attention', [64, 20],
                                                                              4,
                                                                              activation='silu')(
        [caixi_short_order_values_seqs_embs,
         caixi_short_order_keys_seqs_embs,
         short_order_querys_concat,
         order_seq_mask_caixi_short]
    )
    print("caixi short下单的长度：" + str(caixi_short_order_attention_tensor_long))
    # ~------------- original long/short click/order seq process
    click_keys_seqs_infos = [
        'f_click_freq_seq_week_20',
        'f_click_freq_seq_hour_20',
        'f_click_freq_seq_timegap_20'
    ]
    click_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                       [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                            for namestr in click_keys_seqs_infos]
    click_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_click_freq_seq_dtype_20'],
                                        [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    order_keys_seqs_infos = [
        'f_order_freq_seq_week_20',
        'f_order_freq_seq_hour_20',
        'f_order_freq_seq_timegap_20'
    ]
    order_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                       [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                            for namestr in order_keys_seqs_infos]
    order_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_order_freq_seq_dtype_20'],
                                        [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    # --
    ori_click_attention_tensor_long = ContextualMultiHeadMulAttention('ori_click_attention', [64, 20], 4,
                                                                      activation='silu')(
        [click_values_seqs_embs,
         click_keys_seqs_embs,
         long_click_querys_concat,
         click_seq_mask_long]
    )
    print("ori long点击的长度" + str(ori_click_attention_tensor_long))
    ori_order_attention_tensor_long = ContextualMultiHeadMulAttention('ori_order_attention', [64, 20], 4,
                                                                      activation='silu')(
        [order_values_seqs_embs,
         order_keys_seqs_embs,
         long_order_querys_concat,
         order_seq_mask_long]
    )
    print("ori long下单的长度：" + str(ori_order_attention_tensor_long))
    # origin short
    short_click_keys_seqs_infos = [
        'f_click_weekseq_20',
        'f_click_hourseq_20',
        'f_click_gapseq_20'
    ]
    short_click_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                             [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                  for namestr in short_click_keys_seqs_infos]
    short_click_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_click_dtypeseq_20'],
                                              [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    short_order_keys_seqs_infos = [
        "f_order_weekseq_20",
        "f_order_hourseq_20",
        "f_order_gapseq_20"
    ]
    short_order_keys_seqs_embs = [tf.reshape(namestr2tensor_seq_mapping[namestr],
                                             [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
                                  for namestr in short_order_keys_seqs_infos]
    short_order_values_seqs_embs = tf.reshape(namestr2tensor_seq_mapping['f_order_dtypeseq_20'],
                                              [-1, conf['sparse']['seq_max_len'], conf['sparse']['seq_emb_len']])
    # --
    ori_short_click_attention_tensor_long = ContextualMultiHeadMulAttention('ori_short_click_attention', [64, 20], 4,
                                                                            activation='silu')(
        [short_click_values_seqs_embs,
         short_click_keys_seqs_embs,
         short_click_querys_concat,
         click_seq_mask_short]
    )
    print("short点击的长度：" + str(ori_short_click_attention_tensor_long))
    ori_short_order_attention_tensor_long = ContextualMultiHeadMulAttention('ori_short_order_attention', [64, 20], 4,
                                                                            activation='silu')(
        [short_order_values_seqs_embs,
         short_order_keys_seqs_embs,
         short_order_querys_concat,
         order_seq_mask_short]
    )
    print("short下单的长度：" + str(ori_short_order_attention_tensor_long))
    # 9：顶层网络 把所有的交叉全部concat起来。经过dnn和cgc输出两个任务的向量。gate_input和场景一起作为新的门。
    user_long_action_list = [
        ori_click_attention_tensor_long, caixi_click_attention_tensor_long,
        ori_order_attention_tensor_long, caixi_order_attention_tensor_long,
    ]
    user_short_action_list = [
        ori_short_click_attention_tensor_long, caixi_short_click_attention_tensor_long,
        ori_short_order_attention_tensor_long, caixi_short_order_attention_tensor_long,
    ]
    user_action_list = user_long_action_list + user_short_action_list + [life_out_vector]

    embeds_to_concat = [hour_splits[0], week_splits[0]] + common_embeds + \
                       [sz_session_0_list_sum, sz_session_1_list_sum,
                        sz_session_3_list_sum, sz_session_5_list_sum,
                        user_displayed_item_nums, user_displayed_item_ids, i2i_attention_tensor]  #
    # 3：fm部分，uid与iid内积/iid与geo内积/iid分别与user行为内积。
    f_uid_emb = namestr2tensor_embed_mapping['f_user_id']
    f_iid_emb = namestr2tensor_embed_mapping['f_item_type_id']
    f_u_geohash_emb = namestr2tensor_embed_mapping['f_geohash']
    user_iids_features = ['f_user_view_item_dtype_lists_6h', 'f_user_view_item_dtype_lists_24h',
                          'f_user_view_item_dtype_lists',
                          'f_user_order_item_dtype_lists']
    user_iids_lists = [namestr2tensor_embed_mapping[x] for x in user_iids_features]
    print('3：fm-part-len user iids lists : ' + str(len(user_iids_lists)))
    def add_fm(l_embed, r_embeds, outputs):
        for r_embed in r_embeds:
            r = tf.reduce_sum(
                tf.multiply(l_embed, r_embed), 1, keepdims=True
            )
            outputs.append(r)
    fm_outputs = []
    add_fm(f_uid_emb, [f_iid_emb], fm_outputs)
    add_fm(f_iid_emb, [f_u_geohash_emb] + user_iids_lists, fm_outputs)
    # 4: mix部分 直接concat。门部分提取出来。
    mix_feature_embeds = get_my_collection('mix_feature_embeds')
    # 5: 生成 mix_mask
    buz_kind_size = 8
    mix_buz_index = tf.gather(dense_input, [145], axis=1)
    mix_buz_index = tf.reduce_sum(mix_buz_index, axis=1)
    mix_buz_index = buz_kind_size - mix_buz_index
    zeros = tf.zeros_like(mix_buz_index)
    mix_buz_index = tf.where(mix_buz_index < buz_kind_size, x=mix_buz_index, y=zeros)
    mix_buz_index = tf.cast(mix_buz_index, dtype=tf.int32)
    batch_size = tf.shape(mix_buz_index)[0]
    mix_batch_index = tf.range(batch_size)
    mix_index = tf.stack([mix_batch_index, mix_buz_index], axis=1)
    mix_mask_base = tf.zeros([batch_size, buz_kind_size], dtype=tf.int32)
    mix_mask_ones = tf.ones([batch_size], dtype=tf.int32)
    x_shape = tf.shape(mix_mask_base)
    mix_mask = tf.scatter_nd(mix_index, mix_mask_ones, x_shape)
    mix_mask = tf.expand_dims(mix_mask, axis=-1)
    mix_mask = tf.cast(mix_mask, tf.float32)
    def mask_by_buz(in_tensor, how='tile', share_size=0):
        if how == 'tile':
            ori_size_ = int(in_tensor.shape[-1])
            mix_n_ = tf.tile(in_tensor, [1, buz_kind_size])
        elif how == 'split':
            ori_size_ = int(in_tensor.shape[-1]) // buz_kind_size
            assert int(in_tensor.shape[-1]) == ori_size_ * buz_kind_size
            mix_n_ = in_tensor
        mix_n_ = tf.reshape(mix_n_, [-1, buz_kind_size, ori_size_])
        mix_v_ = mix_n_ * mix_mask
        mix_v_ = EvoNormNew(prefix='buz_norm', is_training=is_training, last=True)(mix_v_)
        mix_v_ = tf.reshape(mix_v_, [-1, buz_kind_size * ori_size_])
        return mix_v_
    mix_feature_input = tf.concat([mix_feature_embeds[0], mix_feature_embeds[1]], axis=1)
    mix_v = mask_by_buz(mix_feature_input, how='tile')
    dense_input = tf.gather(dense_input, (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,
                                          22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44,
                                          45, 46, 47, 48, 49, 50, 51, 52, 53,
                                          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                                          73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                                          84, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104,
                                          105, 106, 107, 109, 110, 111, 112,
                                          113, 114, 115, 116, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129,
                                          130, 131, 132, 133, 134, 136, 137, 138,
                                          139, 141, 143), axis=1)
    print("dense删除完了的大小：" + str(dense_input))
    dense_in = add_input_layer(dense_input)
    concat_in = tf.concat(embeds_to_concat, axis=1)

    ## 交叉网络
    mix_out = DNN(hidden_units=[384, 192], name='mix_out_dnn', is_training=is_training, activation='evonorm')(
        mix_v)
    dense_out = DNN(hidden_units=[384, 192], name='dense_output_dnn', is_training=is_training, activation='dice')(
        dense_in)
    # concat_out = DNN(hidden_units=[384, 192], name='concat_out_dnn', is_training=is_training, activation='dice')(
    #    concat_in)
    # user_action_in = tf.concat(user_action_list, axis=-1)
    # user_action_out = DNN(hidden_units=[384, 192], name='user_action_out_dnn', is_training=is_training, activation='dice')(
    #     user_action_in)
    # i2i_attention_out = DNN(hidden_units=[384, 192], name='i2i_action_out_dnn', is_training=is_training, activation='dice')(
    #     i2i_attention_tensor)
    with tf.name_scope("cross_net"):
        # mix_cross_dense = mix_out * dense_out
        # mix_cross_concat = mix_out * concat_out
        # mix_cross_user = mix_out * user_action_out


        with tf.variable_scope("item_cross_scene"):
            with tf.variable_scope("scene", reuse=tf.AUTO_REUSE):
                scene_embeds = get_my_collection('scene_embeds')
                dnn_input = tf.concat([hour_splits[1], week_splits[1]] + scene_embeds, 1)
                scene_vector = DNN(hidden_units=[64, 32], name='scene_dnn2', is_training=is_training, activation='dice')(
                    dnn_input)
            item_info_emb = get_my_collection('item_info_embeds')
            item_emb = tf.concat(item_info_emb, 1)
            item_out = DNN(hidden_units=[32], name='item_map_16dim', is_training=is_training, activation='dice')(
                item_emb)
            item_cross_scene = item_out * scene_vector
        cross_out = tf.concat([item_cross_scene], axis=1)

    concat_list = fm_outputs + [concat_in, mix_out, dense_out] + [cross_out] + user_action_list

    fc_input = tf.concat(concat_list, 1, name='MLP_INPUT')
    print("原top-2000-现在多少: " + str(fc_input))
    this_dim = int(fc_input.shape[-1])
    with tf.variable_scope("dnn_common"):

        fc_input_1 = DNN(hidden_units=[this_dim, this_dim, 512], name='dnn_common', is_training=is_training, activation='dice')(fc_input)
        fc_input_ctr = DNN(
            hidden_units=[256, 128],
            name='dnn_ctr',
            is_training=is_training,
            activation='dice',
            output_activation='evonorm',
        )(fc_input_1)
        fc_input_cvr = DNN(
            hidden_units=[256, 128],
            name='dnn_cvr',
            is_training=is_training,
            activation='dice',
            output_activation='evonorm',
        )(fc_input_1)
    print("cgc output click task:" + str(fc_input_ctr))
    print("cgc output order task:" + str(fc_input_cvr))
    # 10：线性部分和fm部分一起走了tanh，作为新的线性部分
    print("原线性部分：" + str(linear))
    linear_output = tf.concat([linear] + fm_outputs, 1)
    linear_output = tf.nn.tanh(linear_output)
    print("原线性+fm部分-新线性：" + str(linear_output))
    # 11：两个任务，线性fc-64 + taskfc-64 -》add-64 -》relu -》 fc-1

    with tf.variable_scope("dnn_click_task"):
        linear_input_ctr = add_fc(linear_output, 64, None, 'linear_ctr')
        fc_input_ctr_cas = add_fc(fc_input_ctr, 64, None, 'fc1')
        fc_input_ctr = tf.nn.relu(tf.add_n([fc_input_ctr_cas, linear_input_ctr]))
        fc_output_ctr = add_fc(fc_input_ctr, 1, None, 'fc_out')
    with tf.variable_scope("dnn_order_task"):
        linear_input_cvr = add_fc(linear_output, 64, None, 'linear_cvr')
        fc_input_cvr = add_fc(fc_input_cvr, 64, None, 'fc1')
        # fc_input_cvr = add_fc(tf.concat([fc_input_cvr, fc_input_ctr_cas], axis=-1), 64, None, 'fc2')
        fc_input_cvr = tf.nn.relu(tf.add_n([fc_input_cvr, linear_input_cvr]))
        fc_output_cvr = add_fc(fc_input_cvr, 1, None, 'fc_out')

    y_ctr = tf.sigmoid(fc_output_ctr)
    y_ctcvr = tf.sigmoid(fc_output_cvr)

    def aux_support(fc_input, linear_output, layer_size=[1024], name="aux_task"):
        with tf.variable_scope(name+"_aux_task"):
            fc_input_aux = DNN(hidden_units=layer_size, name=name+'_dnn_common', is_training=is_training, activation='dice')(fc_input)
            fc_input_aux_tower1 = DNN(
                hidden_units=[512,256,128],
                name=name+'_dnn_ctr',
                is_training=is_training,
                activation='dice',
                output_activation='evonorm',
            )(fc_input_aux)
            fc_input_aux_tower2 = DNN(
                hidden_units=[512,256,128,128],
                name=name+'_dnn_ctr',
                is_training=is_training,
                activation='dice',
                output_activation='evonorm',
            )(fc_input_aux)
            fc_input_aux_merge = fc_input_aux_tower2 * fc_input_aux_tower1

            linear_input_aux = add_fc(linear_output, 64, None, 'linear_cvr')
            fc_input_aux = add_fc(fc_input_aux_merge, 64, None, 'fc1')

            fc_input_aux = tf.nn.relu(tf.add_n([fc_input_aux , linear_input_aux]))
            fc_output_aux = add_fc(fc_input_aux, 1, None, 'fc_out')
            fc_output_aux = tf.sigmoid(fc_output_aux)
            fc_output_aux_stop_gradient = tf.stop_gradient(fc_output_aux*1.0)
        return fc_output_aux,fc_output_aux_stop_gradient

    fc_output_aux_ctr, fc_output_aux_stop_gradient_ctr = aux_support(fc_input,linear_output,layer_size=[this_dim,this_dim],name="ctr_main")
    fc_output_aux_ctcvr, fc_output_aux_stop_gradient_ctcvr = aux_support(fc_input,linear_output,layer_size=[this_dim,this_dim],name="ctcvr_main")

    loss_ctr_aux = tf.losses.log_loss(
        label_ctr, fc_output_aux_ctr, sample_weight_ctr,epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    loss_ctcvr_aux = tf.losses.log_loss(
        label_ctcvr, fc_output_aux_ctcvr, sample_weight_ctcvr,epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    loss_aux  = loss_ctcvr_aux + loss_ctr_aux

    loss_ctr = tf.losses.log_loss(
        label_ctr, y_ctr, sample_weight_ctr, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)
    loss_ctcvr = tf.losses.log_loss(
        label_ctcvr, y_ctcvr, sample_weight_ctcvr, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    loss_rd_ctr = tf.losses.log_loss(fc_output_aux_stop_gradient_ctr, y_ctr, sample_weight_ctr, epsilon=1e-10,
                                     reduction=tf.losses.Reduction.SUM)
    loss_rd_ctcvr = tf.losses.log_loss(fc_output_aux_stop_gradient_ctcvr, y_ctcvr, sample_weight_ctcvr, epsilon=1e-10,
                                       reduction=tf.losses.Reduction.SUM)
    loss_ctr = loss_ctr + 0.2*loss_rd_ctr
    loss_ctcvr = loss_ctcvr + 0.2*loss_rd_ctcvr

    graph_state = kwargs.get('graph_state', None)
    if graph_state == graph_state_helper('PREDICT', 'rank_online_serving'):
        return [
            (None, y_ctr, None, None),
            (None, y_ctcvr, None, None)
        ]
    return [
        (loss_ctr, y_ctr, label_ctr, sample_weight_ctr),
        (loss_ctcvr, y_ctcvr, label_ctcvr, sample_weight_ctcvr),
        (loss_aux, y_ctcvr, label_aux, sample_weight_aux)
    ]
def main():
    m = build_model()
if __name__ == '__main__':
    main()