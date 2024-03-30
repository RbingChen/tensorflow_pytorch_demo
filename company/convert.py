import os
import sys
sys.path.append('proto/gen-py')

from google.protobuf import text_format
from ps_pb2 import DumpMeta
import tensorflow as tf

def mkdir(dir_path):
    dir_path = dir_path.strip()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_graph(mlx_graph_def, model_name):
    save_path = model_name + '/' + model_name + '.graph.pb'
    with open(save_path, 'wb') as graph_file:
        graph_file.write(mlx_graph_def)
    print 'saved graph: %s' % save_path


def save_input(mlx_graph_def, model_name):
    meta_graph = tf.MetaGraphDef()
    meta_graph.ParseFromString(mlx_graph_def)
    save_path = model_name + '/' + model_name + '.input.txt'
    with open(save_path, 'wb') as input_file:
        input_file.write('// feature inputs as following:' + '\r\n')
        input_file.write('tf_placeholder_dense_cols' + '\r\n' + 'tf_placeholder_linear_cols' + '\r\n')
        for node in meta_graph.graph_def.node:
            if node.op == 'Placeholder' and (node.name.startswith('tf_placeholder_f_')
                                             or node.name.startswith('tf_placeholder_v2_')):
                input_file.write(node.name + '\r\n')

        input_file.write('\r\n' + '// weight inputs as following:' + '\r\n')
        for node in meta_graph.graph_def.node:
            if node.op == 'Assign' and node.name.startswith('Assign'):
                input_file.write(node.name + ':0' + '\r\n')
    print 'saved inputs: %s' % save_path


def convert_graph(model_name, meta_file_path, tf_phase='tf_backend_2'):
    with open(meta_file_path, 'rb') as meta_file:
        dump_meta = DumpMeta()
        text_format.Parse(meta_file.read(), dump_meta)
        model_def = dump_meta.model_defs[0]
        mkdir(model_name)
        for op in model_def.graph.ops:
            if op.name.startswith(tf_phase):
                graph_def = op.tf_backend_attr.graph_attrs[0].graph_def
                save_graph(graph_def, model_name)
                save_input(graph_def, model_name)


if __name__ == '__main__':
    input_model_name = sys.argv[1]
    input_meta_path = sys.argv[2]
    convert_graph(input_model_name, input_meta_path)