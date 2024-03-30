#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
from google.protobuf import text_format

from mlx.python.util.graph_state_helper import graph_state_helper
from features_life_ec_v3_152_newseq import *
from layers import *

global g_mlx_feature_names, g_mlx_embed_names, g_mlx_embed_mapping
global g_mlx_seq_feature_names, g_mlx_embed_seq_mapping, g_mlx_embed_mask_mapping
global g_my_name_collections, g_seq_embed_mapping, g_seq_embed_names


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
    global g_mlx_feature_names, g_mlx_embed_names, g_mlx_embed_mapping
    global g_mlx_seq_feature_names, g_mlx_embed_seq_mapping, g_mlx_embed_mask_mapping
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
    is_training = True
    # model函数的重点是传递tensor！！！
    model(dense_input,
          linear_output,
          [g_mlx_embed_mapping[x] for x in g_mlx_feature_names],  # 必须保证顺序！保存的是tensor
          [g_mlx_embed_seq_mapping[x] for x in g_mlx_seq_feature_names],
          [g_mlx_embed_mask_mapping[x] for x in g_mlx_seq_feature_names],
          label_ctr,
          label_ctcvr,
          sample_weight_ctr,
          sample_weight_ctcvr,
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


extra_graph_states = [
    graph_state_helper('TRAIN'),
    graph_state_helper('EVALUATE'),
    graph_state_helper('PREDICT', 'default'),
    graph_state_helper('PREDICT', 'top_fc_precompute')
]


@tfmlx.tf_wrapper(extra_graph_states, no_default_states=True)
def model(dense_input, linear, embeddings, seq_embeddings, seq_embeddings_mask, label_ctr, label_ctcvr,
          sample_weight_ctr,
          sample_weight_ctcvr, is_training=True, **kwargs):
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
    hour = namestr2tensor_embed_mapping["f_hour"]
    week = namestr2tensor_embed_mapping["f_weekday"]

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

    hour_week_concat = tf.concat([week, hour], axis=-1)

    # i2i network
    SEQ_MAX_LEN = 20

    def reshape_seq(seq, max_len=SEQ_MAX_LEN):
        return [tf.reshape(x, [-1, max_len, x.shape[-1] // max_len]) for x in seq]

    i2i_seq_embeds_list = reshape_seq(get_emb_seq_collection('i2i_seq_embeds'))
    i2i_seq_mask = get_mask_seq_collection('i2i_seq_embeds')[0]
    if len(i2i_seq_embeds_list) > 1:
        i2i_seq_embeds = tf.concat(i2i_seq_embeds_list, axis=-1)
    else:
        i2i_seq_embeds = i2i_seq_embeds_list[0]
    i2i_querys_concat = tf.tile(tf.expand_dims(hour_week_concat, 1), [1, SEQ_MAX_LEN, 1])
    i2i_seq_emb_size = int(i2i_seq_embeds.shape[-1])
    i2i_seq_mask = tf.expand_dims(i2i_seq_mask, axis=-1)
    i2i_seq_mask = tf.tile(i2i_seq_mask, [1, 1, i2i_seq_emb_size])

    i2i_attention_tensor = ContextualMultiHeadMulAttentionNew(
        'i2i_attention',
        [64, i2i_seq_emb_size],
        2,
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
    # life_item_embed_to_query = tf.concat(life_item_embeds,axis = -1)

    ## life_seq
    life_seq_max_len = 100
    life_seq_embeds_list = reshape_seq(get_emb_seq_collection('life_seq_embeds'), life_seq_max_len)
    life_seq_mask = get_mask_seq_collection('life_seq_embeds')[0]
    life_seq_embeds = tf.concat(life_seq_embeds_list, axis=-1)

    life_seq_embeds = tf.split(life_seq_embeds, [20, 80], axis=1)[0]

    life_seq_mask = tf.split(life_seq_mask, [20, 80], axis=1)[0]

    item_info_emb = get_my_collection('item_info_embeds')
    item_info_embs = tf.concat(life_item_embeds + item_info_emb, axis=-1)

    life_seq_emb_size = int(life_seq_embeds.shape[-1])
    life_attention_tensor = ContextualMultiHeadMulAttention2('life_attention', [64, life_seq_emb_size], 2,
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

    hour_week_context = tf.tile(tf.expand_dims(hour_week_concat, 1), [1, conf['sparse']['seq_emb_len'], 1])

    def reshape_seq_by_names(names, max_len=SEQ_MAX_LEN, return_list=False):
        this_seq_list = [namestr2tensor_seq_mapping[namestr] for namestr in names]
        res = reshape_seq(this_seq_list, max_len)
        if not return_list:
            return tf.concat(res, axis=-1)

    seq_input_list = [
        [
            [
                'f_guessclick_freq_seq_dtype_20'
            ],
            [
                'f_guessclick_freq_seq_week_20',
                'f_guessclick_freq_seq_hour_20',
                'f_guessclick_freq_seq_timegap_20',
            ],
            None,
            click_seq_mask_caixi,
        ],
        [
            [
                'f_guessorder_freq_seq_dtype_20'
            ],
            [
                'f_guessorder_freq_seq_week_20',
                'f_guessorder_freq_seq_hour_20',
                'f_guessorder_freq_seq_timegap_20',
            ],
            None,
            order_seq_mask_caixi,
        ],
        [
            [
                'f_guessclick_dtypeseq_20',
            ],
            [
                'f_guessclick_weekseq_20',
                'f_guessclick_hourseq_20',
                'f_guessclick_gapseq_20',
            ],
            None,
            click_seq_mask_caixi_short,
        ],
        [
            [
                'f_guessorder_dtypeseq_20',
            ],
            [
                "f_guessorder_weekseq_20",
                "f_guessorder_hourseq_20",
                "f_guessorder_gapseq_20",
            ],
            None,
            order_seq_mask_caixi_short,
        ],
        [
            [
                'f_click_freq_seq_dtype_20',
            ],
            [
                'f_click_freq_seq_week_20',
                'f_click_freq_seq_hour_20',
                'f_click_freq_seq_timegap_20'
            ],
            None,
            click_seq_mask_long,
        ],
        [
            [
                'f_order_freq_seq_dtype_20',
            ],
            [
                'f_order_freq_seq_week_20',
                'f_order_freq_seq_hour_20',
                'f_order_freq_seq_timegap_20',
            ],
            None,
            order_seq_mask_long,
        ],
        [
            [
                'f_click_dtypeseq_20',
            ],
            [
                'f_click_weekseq_20',
                'f_click_hourseq_20',
                'f_click_gapseq_20',
            ],
            None,
            click_seq_mask_short,
        ],
        [
            [
                'f_order_dtypeseq_20',
            ],
            [
                "f_order_weekseq_20",
                "f_order_hourseq_20",
                "f_order_gapseq_20",
            ],
            None,
            order_seq_mask_short,
        ],
    ]

    key_list = []
    query_list = []
    context_list = [hour_week_context]
    mask_list = []

    for k, q, c, m in seq_input_list:
        new_k = reshape_seq_by_names(k, return_list=False)
        key_list.append(new_k)
        new_q = reshape_seq_by_names(q, return_list=False)
        query_list.append(new_q)
        if c is not None:
            context_list.append(c)
        new_m = m
        if len(m.shape) != len(new_k.shape):
            new_m = tf.expand_dims(m, axis=-1)
            new_m = tf.tile(new_m, [1, 1, int(new_k.shape[-1])])
        mask_list.append(new_m)

    seq_key = tf.concat(key_list, axis=-1)
    seq_query = tf.concat(query_list, axis=-1)
    seq_context = tf.concat(context_list, axis=-1)
    seq_mask = tf.concat(mask_list, axis=-1)

    seq_input = [
        seq_key,
        seq_query,
        seq_context,
        seq_mask,
    ]

    seq_mid_dim = int(seq_query.shape[-1]) + int(seq_context.shape[-1])
    seq_out_dim = int(seq_key.shape[-1])

    seq_output = ContextualMultiHeadMulAttentionNew(
        'seq_all_attention',
        [seq_mid_dim, seq_out_dim],
        2,
        activation='silu',
    )(seq_input)

    user_action_tensor = tf.concat([seq_output, life_out_vector], axis=-1)

    embeds_to_concat = [hour_week_concat] + common_embeds + \
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
    tag = mix_buz_index
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
    mix_mask_h = tf.abs(mix_mask - 1.0)
    mix_mask_h = tf.cast(mix_mask_h, tf.float32)
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
        # mix_v_ = EvoNormNew(prefix='buz_norm', is_training=is_training, last=True)(mix_v_)
        mix_v_ = tf.reshape(mix_v_, [-1, buz_kind_size * ori_size_])
        return mix_v_

    mix_feature_input = tf.concat([mix_feature_embeds[0], mix_feature_embeds[1]], axis=1)
    mix_v = mask_by_buz(mix_feature_input, how='tile')

    user_type = tf.gather(dense_input, [151], axis=1)

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
    dense_in = add_input_layer(dense_input)
    dense_in = tf.add_n([dense_in + dense_input])
    concat_in = tf.concat(embeds_to_concat, axis=1)

    ## 交叉网络

    mix_out = DNN(hidden_units=[384, 192], name='mix_out_dnn', is_training=is_training, activation='evonorm')(
        mix_v)
    dense_out = DNN(hidden_units=[384, 192], name='dense_output_dnn', is_training=is_training, activation='dice')(
        dense_in)

    # concat_out = DNN(hidden_units=[384, 192], name='concat_out_dnn', is_training=is_training, activation='dice')(
    #    concat_in)
    user_action_in = user_action_tensor
    user_action_out = DNN(hidden_units=[384, 192], name='user_action_out_dnn', is_training=is_training,
                          activation='dice')(
        user_action_in)
    with tf.name_scope("cross_net"):
        mix_cross_dense = mix_out * dense_out
        # mix_cross_concat = mix_out * concat_out
        mix_cross_user = mix_out * user_action_out

        with tf.variable_scope("item_cross_scene"):
            with tf.variable_scope("scene", reuse=tf.AUTO_REUSE):
                scene_embeds = get_my_collection('scene_embeds')
                scene_emb = tf.concat(scene_embeds,axis = -1)
                dnn_input = tf.concat([hour_week_concat] + scene_embeds, 1)
                scene_vector = DNN(hidden_units=[64, 32], name='scene_dnn2', is_training=is_training,
                                   activation='dice')(
                    dnn_input)
            item_info_emb = get_my_collection('item_info_embeds')
            item_emb = tf.concat(item_info_emb, 1)
            item_out = DNN(hidden_units=[32], name='item_map_16dim', is_training=is_training, activation='dice')(
                item_emb)
            item_cross_scene = item_out * scene_vector
        cross_out = tf.concat([mix_cross_dense, item_cross_scene, mix_cross_user], axis=1)

    concat_list = fm_outputs + [concat_in, mix_out, dense_out,item_emb,scene_emb] + [cross_out] + [user_action_tensor]

    fc_input = tf.concat(concat_list, 1, name='MLP_INPUT')
    print("原top-2000-现在多少: " + str(fc_input))

    mix_buz_index_up = tf.squeeze(tag, [1])
    mix_buz_index_up = tf.cast(mix_buz_index_up, dtype=tf.int32)
    buz_zeros = tf.zeros_like(mix_buz_index_up)
    buz_ones = tf.ones_like(mix_buz_index_up)
    ## 原始编码，8个塔
    buz_waimai_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 1), buz_ones, buz_zeros)
    buz_deal_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 2), buz_ones, buz_zeros)
    buz_product_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 3), buz_ones, buz_zeros)
    buz_content_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 4), buz_ones, buz_zeros)
    buz_ecommerce_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 5), buz_ones, buz_zeros)
    buz_jiulv_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 6), buz_ones, buz_zeros)
    buz_daozong_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 7), buz_ones, buz_zeros)
    buz_others_ = tf.where(tf.equal(mix_buz_index_up, buz_ones * 8), buz_ones, buz_zeros)
    ## 重新编码，4个塔：电商、外卖、到综、内容
    buz_dianshang = tf.add_n([buz_ecommerce_ * 0, buz_others_ * 0])
    buz_waimai = buz_waimai_  # 1
    buz_daozong = tf.add_n([buz_deal_ * 2, buz_product_ * 2, buz_jiulv_ * 2, buz_daozong_ * 2])
    buz_content = buz_content_ * 3
    ## 重编码后的4塔索引：0,1,2,3
    buz_index = tf.add_n([buz_dianshang, buz_waimai, buz_daozong, buz_content])
    buz_cnt = 4
    ## 对业务索引进onehot编码，生成mask
    buz_mask = tf.one_hot(buz_index, depth=buz_cnt, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
    print("buz_mask",str(buz_mask))
    #mix_mask = tf.expand_dims(buz_mask, axis=-1)

    def PPNet(common_emb,item_emb,name):
        common_emb = tf.layers.batch_normalization(
            inputs=common_emb,
            axis=-1,
            epsilon=1e-9,
            center=True,
            scale=True,
            training=is_training)
        p_dnn_embed = DNN(hidden_units=[32],name='dnn_pp_{}'.format(name),is_training=is_training,activation='dice',output_activation='dice')(common_emb)
        p_dnn_gate = DNN(hidden_units=[32],name='dnn_pp_gate_{}'.format(name),is_training=is_training,activation='dice',output_activation='dice')(item_emb)
        p_dnn_merge = p_dnn_embed * p_dnn_gate
        p_dnn_merge = tf.add_n([p_dnn_merge,p_dnn_embed])
        p_dnn_merge = add_fc(p_dnn_merge, 1, None, name)
        return p_dnn_merge


    with tf.variable_scope("dnn_common"):
        this_dim = 2048
        fc_input = DNN(hidden_units=[this_dim, this_dim, 512], name='dnn_common', is_training=is_training,
                       activation='dice')(fc_input)
        fc_input_ctr = DNN(
            hidden_units=[256, 128],
            name='dnn_ctr',
            is_training=is_training,
            activation='dice',
            output_activation='evonorm',
        )(fc_input)
        fc_input_cvr = DNN(
            hidden_units=[256, 128],
            name='dnn_cvr',
            is_training=is_training,
            activation='dice',
            output_activation='evonorm',
        )(fc_input)
    # 10：线性部分和fm部分一起走了tanh，作为新的线性部分
    linear_output = tf.concat([linear] + fm_outputs, 1)
    linear_output = tf.nn.tanh(linear_output)
    # 11：两个任务，线性fc-64 + taskfc-64 -》add-64 -》relu -》 fc-1
    with tf.variable_scope("multi-business_gates"):
        gate_vector = tf.concat([mix_feature_embeds[2]], 1)
        gate_vector = DNN(hidden_units=[20], name='gate20', is_training=is_training, activation='dice')(gate_vector)
        multi_tower_logits = add_fc(gate_vector, buz_cnt, None, 'weight') # b,4
        multi_tower_weights = tf.nn.softmax(multi_tower_logits)
        fusion_emb = tf.concat([item_emb],axis = -1)
        print("fusion_emb",str(fusion_emb))

    with tf.variable_scope("dnn_click_task"):
        linear_input_ctr = add_fc(linear_output, 64, None, 'linear_ctr')
        fc_input_ctr_cas = add_fc(fc_input_ctr, 64, None, 'fc1')

        multi_tower_input_ctr = tf.add_n([fc_input_ctr_cas, linear_input_ctr])
        # multi_tower_input_ctr = tf.concat([fc_input_ctr_cas, linear_input_ctr], axis=1)
        fc_output_ctr = []
        for i in range(buz_cnt):
            tower_output_layer1 = PPNet(multi_tower_input_ctr, fusion_emb,'fc_layer1{}'.format(i))
            fc_output_ctr.append(tower_output_layer1) # list 8个 每个 b,1
        fc_output_ctr = tf.stack(fc_output_ctr,axis = 1) # b,8,1
        print("fc_output_ctr",str(fc_output_ctr))
        tower_else = tf.expand_dims(multi_tower_weights,axis = -1) * fc_output_ctr
        tower_else = tf.reduce_sum(tower_else,axis = 1)
        fc_output_ctr = tower_else

    with tf.variable_scope("dnn_order_task"):
        linear_input_cvr = add_fc(linear_output, 64, None, 'linear_cvr')
        fc_input_cvr = add_fc(fc_input_cvr, 64, None, 'fc1')
        fc_input_cvr = tf.nn.relu(tf.add_n([fc_input_cvr, linear_input_cvr]))
        fc_output_cvr = add_fc(fc_input_cvr, 1, None, 'fc_out')
    y_ctr = tf.sigmoid(fc_output_ctr)
    y_ctcvr = tf.sigmoid(fc_output_cvr)
    loss_ctr = tf.losses.log_loss(
        label_ctr, y_ctr, sample_weight_ctr, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    loss_buz = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = multi_tower_logits , labels = tf.stop_gradient(buz_mask)))

    loss_ctcvr = tf.losses.log_loss(
        label_ctcvr, y_ctcvr, sample_weight_ctr, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    user_emb = tf.stop_gradient(tf.identity(f_uid_emb))
    u_emb = tf.nn.l2_normalize(user_emb, axis=1)
    u2u_sim = tf.matmul(u_emb, u_emb, transpose_b=True)
    pair_zeros = tf.zeros_like(u2u_sim)
    pair_ones = tf.ones_like(u2u_sim)
    u2u_sim_weight = tf.where(u2u_sim > 0.99, pair_ones, pair_zeros)

    user_good = user_type * u2u_sim_weight

    true_ctr = tf.cast(label_ctr, y_ctr.dtype)
    true_ctr_t = tf.transpose(true_ctr)
    true_ctr_diff = true_ctr - true_ctr_t
    true_ctr_weight = tf.where(true_ctr_diff > 0.5, pair_ones, pair_zeros)

    h_ctr = fc_output_ctr
    h_ctr_t = tf.transpose(h_ctr)
    h_ctr_diff = h_ctr - h_ctr_t
    y_ctr_diff = tf.sigmoid(h_ctr_diff)

    sample_weight_ctr_pair = true_ctr_weight * user_good
    label_ctr_pair = pair_ones

    pair_loss_ctr = tf.losses.log_loss(
        label_ctr_pair, y_ctr_diff, sample_weight_ctr_pair, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    true_ctcvr = tf.cast(label_ctcvr, y_ctcvr.dtype)
    true_ctcvr_t = tf.transpose(true_ctcvr)
    true_ctcvr_diff = true_ctcvr - true_ctcvr_t
    true_ctcvr_weight = tf.where(true_ctcvr_diff > 0.5, pair_ones, pair_zeros)

    h_ctcvr = fc_output_cvr
    h_ctcvr_t = tf.transpose(h_ctcvr)
    h_ctcvr_diff = h_ctcvr - h_ctcvr_t
    y_ctcvr_diff = tf.sigmoid(h_ctcvr_diff)

    sample_weight_ctcvr_pair = true_ctcvr_weight * user_good
    label_ctcvr_pair = pair_ones

    pair_loss_ctcvr = tf.losses.log_loss(
        label_ctcvr_pair, y_ctcvr_diff, sample_weight_ctcvr_pair, epsilon=1e-10,
        reduction=tf.losses.Reduction.SUM)

    pair_loss = pair_loss_ctr + pair_loss_ctcvr
    pair_loss = pair_loss * 0.1

    return [
        (loss_ctr + loss_ctcvr, y_ctr, label_ctr, sample_weight_ctr),
        (pair_loss + loss_buz, y_ctcvr, label_ctcvr, sample_weight_ctcvr),
    ]


def main():
    m = build_model()

if __name__ == '__main__':
    main()