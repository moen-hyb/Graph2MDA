from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelGAT(Model):
    def __init__(self, placeholders, num_features, features_nonzero, hidden1, hidden2, num_class, num_target,**kwargs):
        super(GCNModelGAT, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = num_target
        self.features_nonzero = features_nonzero
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=self.hidden1_dim,
                                           output_dim=self.hidden2_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=self.hidden2_dim,
                                        act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, hidden1, hidden2, num_class, num_target,**kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.output_dim = num_target
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)


        self.hidden2 = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       adj=self.adj,
                                       act=tf.nn.relu,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        # self.hidden3 = GraphConvolution(input_dim=self.hidden2_dim,
        #                                output_dim=self.hidden2_dim,
        #                                adj=self.adj,
        #                                act=tf.nn.relu,
        #                                dropout=self.dropout,
        #                                logging=self.logging)(self.hidden2)

        self.z_mean = GraphConvolution(input_dim=self.hidden2_dim,
                                       output_dim=self.output_dim,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden2)

        self.z_log_std = GraphConvolution(input_dim=self.hidden2_dim,
                                          output_dim=self.output_dim,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden2)

        self.z = self.z_mean + tf.random_normal([self.n_samples, self.output_dim], dtype=tf.float64) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=self.output_dim,
                                        act=lambda x: x,
                                      logging=self.logging)(self.z)
