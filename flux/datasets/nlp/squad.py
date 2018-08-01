"""
Data download and parsing for the squad dataset
"""

import json

import tensorflow as tf

from flux.backend.globals import DATA_STORE
from flux.backend.datastore import KeyExistsError
from flux.backend.data import maybe_download_and_store_single_file
from flux.processing.nlp.dictionary import NLPDictionary
from flux.processing.nlp.util import get_token_span_from_char_span


class Squad():

    def __init__(self, version: str='2.0') -> None:

        if version == '2.0':
            self.training_data_json_key = maybe_download_and_store_single_file(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json', key='squad/train_json')
            self.dev_data_json_key = maybe_download_and_store_single_file(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', key='squad/dev_json')

            # Load the JSON from the files
            with open(DATA_STORE[self.training_data_json_key], 'r') as train_json:
                self.train_json = json.loads(train_json.read())
            with open(DATA_STORE[self.dev_data_json_key], 'r') as dev_json:
                self.dev_json = json.loads(dev_json.read())

            # Parse the JSON
            
            self.dictionary = NLPDictionary(
                char_maxlen=37, word_maxlen=766, pad_output=True)

            # Build the training set if necessary
            if not DATA_STORE.is_valid('squad/tfrecord/train'):
                num_errors = 0
                num_documents = 0
                print('Building training data...')
                tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key('squad/tfrecord/train', 'data.tfrecords'))
                for idx, article in enumerate(self.train_json['data']):
                    if idx % 1 == 0:
                        print('[{}/{}] Documents Parsed ({} examples, {} errors)'.format(idx, len(self.train_json['data']), num_documents, num_errors))
                    for paragraph_json in article['paragraphs']:

                        # Compute the context embedding
                        context_tokens = self.dictionary.tokenizer.parse(
                            paragraph_json['context'].strip().replace('\n', ''))
                        context_dense = self.dictionary.dense_parse_tokens(
                            context_tokens)

                        # Compute the QA embeddings
                        for question_answer in paragraph_json['qas']:
                            question_dense = self.dictionary.dense_parse(
                                question_answer['question'].strip().replace('\n', ''))

                            for answer in question_answer['answers']:
                                answer_dense = self.dictionary.dense_parse(
                                    answer['text'])

                                # Character span start/end
                                span_start = answer['answer_start']
                                span_end = span_start + len(answer['text'])

                                # Get the token span from the char span
                                token_span_start, token_span_end = get_token_span_from_char_span(
                                    paragraph_json['context'].strip().replace('\n', ''), context_tokens, span_start, span_end)
                                
                                if token_span_start < 0 or token_span_end < 0:
                                    # print('[{}/{}] Error parsing, no token correspondence.'.format(idx, len(self.train_json['data'])))
                                    num_errors += 1
                                    break

                                # Now that we've got the contents, let's make a TF-Record
                                # We're going to handle the tf-record writing here for now
                                # TODO: Move the tf-record writing to it's own file

                                # Create the feature dictionary
                                feature_dict = {}
                                feature_dict['context_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=context_dense[0].flatten()))
                                feature_dict['context_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=context_dense[1].flatten()))
                                feature_dict['question_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=question_dense[0].flatten()))
                                feature_dict['question_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=question_dense[1].flatten()))
                                feature_dict['answer_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=answer_dense[0].flatten()))
                                feature_dict['answer_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=answer_dense[1].flatten()))
                                feature_dict['word_maxlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[766]))
                                feature_dict['char_maxlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[37]))
                                feature_dict['span_start'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[span_start]))
                                feature_dict['span_end'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[span_end]))
                                feature_dict['token_span_start'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[token_span_start]))
                                feature_dict['token_span_end'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[token_span_end]))

                                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                                tf_record_writer.write(example.SerializeToString())
                                num_documents += 1
                tf_record_writer.close()
                DATA_STORE.update_hash('squad/tfrecord/train')

            # Build the validation set if necessary
            if not DATA_STORE.is_valid('squad/tfrecord/dev'):
                num_errors = 0
                num_documents = 0
                print('Building validation data...')
                tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key('squad/tfrecord/dev', 'data.tfrecords'))
                for idx, article in enumerate(self.dev_json['data']):
                    if idx % 1 == 0:
                        print('[{}/{}] Documents Parsed ({} examples, {} errors)'.format(idx, len(self.dev_json['data']), num_documents, num_errors))
                    for paragraph_json in article['paragraphs']:

                        # Compute the context embedding
                        context_tokens = self.dictionary.tokenizer.parse(
                            paragraph_json['context'].strip().replace('\n', ''))
                        context_dense = self.dictionary.dense_parse_tokens(
                            context_tokens)

                        # Compute the QA embeddings
                        for question_answer in paragraph_json['qas']:
                            question_dense = self.dictionary.dense_parse(
                                question_answer['question'].strip().replace('\n', ''))

                            for answer in question_answer['answers']:
                                answer_dense = self.dictionary.dense_parse(
                                    answer['text'])

                                # Character span start/end
                                span_start = answer['answer_start']
                                span_end = span_start + len(answer['text'])

                                # Get the token span from the char span
                                token_span_start, token_span_end = get_token_span_from_char_span(
                                    paragraph_json['context'].strip().replace('\n', ''), context_tokens, span_start, span_end)
                                
                                if token_span_start < 0 or token_span_end < 0:
                                    # print('[{}/{}] Error parsing, no token correspondence.'.format(idx, len(self.dev_json['data'])))
                                    num_errors += 1
                                    break

                                # Now that we've got the contents, let's make a TF-Record
                                # We're going to handle the tf-record writing here for now
                                # TODO: Move the tf-record writing to it's own file

                                # Create the feature dictionary
                                feature_dict = {}
                                feature_dict['context_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=context_dense[0].flatten()))
                                feature_dict['context_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=context_dense[1].flatten()))
                                feature_dict['question_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=question_dense[0].flatten()))
                                feature_dict['question_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=question_dense[1].flatten()))
                                feature_dict['answer_word_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=answer_dense[0].flatten()))
                                feature_dict['answer_char_embedding'] = tf.train.Feature(int64_list=tf.train.Int64List(value=answer_dense[1].flatten()))
                                feature_dict['word_maxlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[766]))
                                feature_dict['char_maxlen'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[37]))
                                feature_dict['span_start'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[span_start]))
                                feature_dict['span_end'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[span_end]))
                                feature_dict['token_span_start'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[token_span_start]))
                                feature_dict['token_span_end'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[token_span_end]))

                                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                                tf_record_writer.write(example.SerializeToString())
                                num_documents +=1
                tf_record_writer.close()
                DATA_STORE.update_hash('squad/tfrecord/dev')

            self.train_fpath = DATA_STORE['squad/tfrecord/train']
            self.dev_fpath = DATA_STORE['squad/tfrecord/dev']

            # Now create the TF-DB which runs the show
            # ?? This is a tomorrow thing :) ??

        else:
            raise NotImplementedError(
                "Only version 2.0 is currently supported")
