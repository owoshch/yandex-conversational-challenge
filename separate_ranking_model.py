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
from model.config import Config
from model.data_utils import minibatches_w_replies, pad_sequences, \
        get_mean_NDCG, rank_predictions, save_submission, load_pairwise_dataset, compute_lengths

from collections import Counter
import tqdm


config = Config()


class RankingModel(BaseModel):
    """Specialized class of Model for NER"""
    

    def __init__(self, config):
        #super(NERModel, self).__init__(config)
        super(RankingModel, self).__init__(config)
        
    def add_placeholders(self):
        
        """Define placeholders = entries to computational graph"""
        
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
                words. 
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
            self.sequence_lengths: sequence_lengths,
            self.replies_lengths: sequence_reply_lengths
        }
        
        if labels is not None:
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
        and we don't train the vectors. 
        """
        
        _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
        
        
        word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")
        
        
        replies_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.reply_word_ids, name="replies_embeddings")
                
        
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        
        self.replies_embeddings =  tf.nn.dropout(replies_embeddings, self.dropout)
        

    
    
    
    def add_logits_op(self):
        """Defines self.logits
        
        For each pair of contexts and reply it returns
        an euclidean distance between mean vector of contexts and 
        mean vector of reply.

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
            
            logits = tf.reduce_mean(logits, axis=1)
            
            
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
            
            
            logits_reply = tf.reduce_mean(logits_reply, axis=1)
            

        self.logits = tf.norm(logits_reply-logits, axis=1, ord='euclidean')
        


    def add_loss_op(self):
        """Defines the loss"""
        
        losses = tf.square(self.labels - self.logits, name="loss")
        self.loss = tf.reduce_mean(losses)
        
        # for tensorboard
        tf.summary.scalar("loss", self.loss)
            
    
        
    def build(self):
        # ranking model specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init
    
    
    
    def predict_batch(self, words, replies):
        """
        Args:
            words: list of sentences
            replies: list of sentences

        Returns:
            labels_pred: list of predicted distances

        """
        fd, sequence_lengths = self.get_feed_dict(words, replies, dropout=1.0)

        labels_pred = self.sess.run(self.logits, feed_dict=fd)

        return labels_pred, sequence_lengths
        
    def predict_proba(self, data, dataframe):
        """
        Args:
            data: list of tuples like (context, reply, distance)
            dataframe: original dataframe to obtain data

        Returns:
            labels_pred: list of predicted distances

        """
        final_predictions = []
        pbar = tqdm.tqdm(total=len(data))
        
        
        distances = compute_lengths(dataframe)
        batch_size = distances[0]
        pbar.update(batch_size)
        for i, (words, replies, labels) in enumerate(minibatches_w_replies(data, batch_size)):
            labels_pred, sequence_lengths = self.predict_batch(words, replies)
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
            test: dataset that yields tuple of (sentences, replies, tags)
            path_to_test: path to dataframe that produces test

        Returns:
            metrics: (dict) metrics["NDCG"] = 76125.4, ...

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
        
        sorted_preds = rank_predictions(test_dataframe, predicted_labels) 
        
    
        test_NDCG = get_mean_NDCG(test_dataframe, sorted_preds)
        
        print ('val NDCG', test_NDCG)

        return {"NDCG": test_NDCG}