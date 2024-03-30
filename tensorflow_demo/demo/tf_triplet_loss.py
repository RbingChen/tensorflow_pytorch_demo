# coding:utf-8

import tensorflow as tf

def get_triplet_mask(labels):
    """
    Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # i, j, k分别是不同的样本索引
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def _pairwise_distances(embeddings, squared=False):
    """
    计算嵌入向量之间的距离
    Args:
        embeddings: 形如(batch_size, embed_dim)的张量
        squared: Boolean. True->欧式距离的平方，False->欧氏距离
    Returns:
        piarwise_distances: 形如(batch_size, batch_size)的张量
    """
    # 嵌入向量点乘，输出shape=(batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # 取dot_product对角线上的值，相当于是每个嵌入向量的L2正则化，shape=(batch_size,)
    square_norm = tf.diag_part(dot_product)

    # 计算距离,shape=(batch_size, batch_size)
    # ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
    # PS: 下面代码计算的是||a - b||^2，结果是一样的
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # 保证距离都>=0
    distances = tf.maximum(distances, 0.0)

    if not squared:
    # 加一个接近0的值，防止求导出现梯度爆炸的情况
        mask = tf.to_float(tf.equal(distances, 0.0))
    distances = distances + mask * 1e-16

    distances = tf.sqrt(distances)

    # 校正距离
    distances = distances * (1.0 - mask)
    return distances

def _get_triplet_mask(labels):
    """
    Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # i, j, k分别是不同的样本索引
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(labels,1)
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    计算整个banch的Triplet loss。
    生成所有合格的triplets样本组，并只对其中>0的部分取均值
    Args:
        labels: 标签，shape=(batch_size,)
        embeddings: 形如(batch_size, embed_dim)的张量
        margin: Triplet loss中的间隔
        squared: Boolean. True->欧氏距离的平方，False->欧氏距离
    Returns:
        triplet_loss: 损失
    """
    # 获取banch中嵌入向量间的距离矩阵
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # 计算一个形如(batch_size, batch_size, batch_size)的3D张量
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # 将invalid Triplet置零
    # label(a) != label(p) or label(a) == label(n) or a == p
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # 删除负值
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # 计算正值
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss


sim = tf.random.uniform(shape=(8,8))

labels = tf.where(sim>0.5,tf.ones_like(sim),tf.zeros_like(sim))
emb = tf.random.uniform(shape=(8,4))
loss = batch_all_triplet_loss(labels,emb,0.01)
with tf.Session() as sess:
    print(sess.run(loss))