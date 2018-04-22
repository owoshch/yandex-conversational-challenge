#imports
import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm

from model.general_utils import Progbar
from model.base_model import BaseModel
from model.matrix_config import Config
from model.data_utils import minibatches, minibatches_w_replies, pad_sequences, \
        load_dataset, load_regression_dataset, one_hot_to_num, \
        get_mean_NDCG, softmax, sort_xgb_predictions, save_submission, load_pairwise_dataset, compute_lengths
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import tqdm


config = Config()


class NERModel(BaseModel):
    """Specialized class of Model for NER"""
    

    def __init__(self, config):
        #super(NERModel, self).__init__(config)
        super(NERModel, self).__init__(config)
        
    def add_placeholders(self):
        
        """Define placeholders = entries to computational graph"""
        #self._word_embeddings = tf.placeholder(dtype=tf.float32,
        #                         shape=[self.config.nwords, self.config.dim_word]) #(dictionary size, embedding_size)
        
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")
        
        
        self.reply_word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")
        
        
        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")
        
        self.replies_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")
        
        
        # shape = (batch size, labels size)
        #self.labels = tf.placeholder(tf.int32, shape=[None, self.config.ntags],
        #                name="labels")
        
        self.labels = tf.placeholder(tf.float32, shape=[None,],
                        name="labels")
        
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")
        
    def get_feed_dict(self, words, replies=None, labels=None, lr=None, dropout=None):
        
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        word_ids, sequence_lengths = pad_sequences(words, 0)
        
        word_reply_ids, sequence_reply_lengths = pad_sequences(replies, 0)
        
        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.reply_word_ids: word_reply_ids,
            #self._word_embeddings: self.config.embeddings,
            self.sequence_lengths: sequence_lengths,
            self.replies_lengths: sequence_reply_lengths
        }
        
        if labels is not None:
            #labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels
        
        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout
            
        return feed, sequence_lengths
    
    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        
        _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        #self._word_embeddings, 
                        #name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
        
        
        
        #_word_embeddings = tf.Variable(self._word_embeddings)
        
        
        word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")
        
        
        replies_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.reply_word_ids, name="replies_embeddings")
                
        
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        
        self.replies_embeddings =  tf.nn.dropout(replies_embeddings, self.dropout)
        

    
    
    
    def add_logits_op(self):
        """Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
            
        
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.hidden_dense_dim])

            b = tf.get_variable("b", shape=[self.config.hidden_dense_dim],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            logits = tf.reshape(pred, [-1, nsteps, self.config.hidden_dense_dim])
            
            print ('original logits shape', logits.shape)
            
            transposed_logits = tf.matmul(logits, logits, transpose_a=True)
            
            
            transposed_logits = tf.tanh(transposed_logits)
            
            
            print ('transposed logits shape', transposed_logits.shape)
            
            
        with tf.variable_scope('proj-dense'):
            W_dense_context = tf.get_variable("W_dense_context", dtype=tf.float32,
                    shape=[self.config.hidden_dense_dim, self.config.hidden_dense_dim])

            b_dense_context = tf.get_variable("b_dense_context", shape=[self.config.hidden_dense_dim],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            
            
            transposed_logits = tf.sigmoid(tf.tensordot(transposed_logits, W_dense_context, axes=1) \
                                           + b_dense_context)
            
            
            print ('transposed logits shape', transposed_logits.shape)
    
            
        with tf.variable_scope("bi-lstm-replies"):
            cell_fw_reply = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw_reply = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw_reply, output_bw_reply), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw_reply, cell_bw_reply, self.replies_embeddings,
                    sequence_length=self.replies_lengths, dtype=tf.float32)
            output_reply = tf.concat([output_fw_reply, output_bw_reply], axis=-1)
            output_reply = tf.nn.dropout(output_reply, self.dropout)
            
        with tf.variable_scope("proj-replies"):
            W_reply = tf.get_variable("W_reply", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.hidden_dense_dim])

            b_reply = tf.get_variable("b_reply", shape=[self.config.hidden_dense_dim],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output_reply)[1]
            output_reply = tf.reshape(output_reply, [-1, 2*self.config.hidden_size_lstm])
            pred_reply = tf.matmul(output_reply, W_reply) + b_reply
            logits_reply = tf.reshape(pred_reply, [-1, nsteps, self.config.hidden_dense_dim])
            
            
            
            print ('original reply logits shape', logits_reply.shape)
            
            #transposed_logits_reply = tf.transpose(logits_reply, perm=[0, 2, 1])
            
            
            
            
            transposed_logits_reply = tf.matmul(logits_reply, logits_reply, transpose_a=True)
            
            print ('transposed reply logits shape', transposed_logits_reply.shape)
            
            transposed_logits_reply = tf.tanh(transposed_logits_reply)
            
            
        with tf.variable_scope('proj-replies-dense'):
            W_dense_reply = tf.get_variable("W_dense_reply", dtype=tf.float32,
                    shape=[self.config.hidden_dense_dim, self.config.hidden_dense_dim])

            b_dense_reply = tf.get_variable("b_dense_reply", shape=[self.config.hidden_dense_dim],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            
            
            transposed_logits_reply = tf.sigmoid(tf.tensordot(transposed_logits_reply, W_dense_reply, axes=1) \
                                                 + b_dense_reply)
            
            print ('transposed reply logits shape', transposed_logits_reply.shape)

        
        #self.logits = correlation
        
        
        self.logits = tf.norm(transposed_logits_reply - transposed_logits, axis=[1, 2], ord='euclidean')
        

        
        
        
            
    
    
    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            #self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
            #        tf.int32)
            self.labels_pred = self.logits
            
            #self.one_hot_labels_pred = self.logits
    

    def add_loss_op(self):
        """Defines the loss"""
        '''
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #        logits=self.logits, labels=self.labels)
        '''
        
        
        losses = tf.square(self.labels - self.logits, name="loss")
        self.loss = tf.reduce_mean(losses)
        
        
        # for tensorboard
        tf.summary.scalar("loss", self.loss)
            
    
        
    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init
    
    
    
    def predict_batch(self, words, replies):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, replies, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths
        
        
    def predict_proba_batch(self, words, replies):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, replies, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
        
            return labels_pred, sequence_lengths


    def predict_proba(self, data, dataframe):
        # data is tuple like while train
        final_predictions = []
        pbar = tqdm.tqdm(total=len(data))
        
        
        distances = compute_lengths(dataframe)
        batch_size = distances[0]
        pbar.update(batch_size)
        for i, (words, replies, labels) in enumerate(minibatches_w_replies(data, batch_size)):
            labels_pred, sequence_lengths = self.predict_proba_batch(words, replies)
            final_predictions.extend(labels_pred)
            batch_size = distances[i + 1]
            pbar.update(batch_size)
        pbar.close()
        return final_predictions
    
    
    
        
    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        #prog = Progbar(target=nbatches)
        
        distances = compute_lengths(pd.read_csv(config.path_to_train))
        batch_size = distances[0]
        
        prog = Progbar(target=len(distances))
        
        # iterate over dataset
        for i, (words, replies, labels) in enumerate(minibatches_w_replies(train, batch_size)):
            fd, _ = self.get_feed_dict(words, replies, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)
            
            #print (i, batch_size)
            batch_size = distances[i + 1]

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["NDCG"] / 100000
    
    def run_evaluate(self, test, path_to_test=None):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        if path_to_test == None:
            path_to_test = self.config.path_to_test
        
        pbar = tqdm.tqdm(total=len(test))
        
        test_dataframe = pd.read_csv(path_to_test)
        test_distances = compute_lengths(test_dataframe)
        batch_size = test_distances[0]
        pbar.update(batch_size)
        predicted_labels = []
        for i, (words, replies, labels) in enumerate(minibatches_w_replies(test, batch_size)):
            labels_pred, sequence_lengths = self.predict_batch(words, replies)
            predicted_labels.extend(labels_pred)
            batch_size = test_distances[i + 1]
            pbar.update(batch_size)
        pbar.close()  
        
        
        print ('my preds', predicted_labels[:5])
        print ('test preds', [x[2] for x in test[:5]])
        sorted_preds = sort_xgb_predictions(test_dataframe, predicted_labels) 
        
    
        test_NDCG = get_mean_NDCG(test_dataframe, sorted_preds)
        
        print ('test NDCG', test_NDCG)
        '''
        val_df = pd.read_csv(self.config.path_to_val)
        val = load_pairwise_dataset(self.config.path_to_val)
        val_preds = self.predict_proba(val, val_df)
        val_NDCG = get_mean_NDCG(val_df, sort_xgb_predictions(val_df, val_preds))
                                 
        print ('val NDCG', val_NDCG)
                                 
        #NDCG = np.mean((test_NDCG, val_NDCG))

        NDCG = val_NDCG
        '''
        return {"NDCG": test_NDCG}