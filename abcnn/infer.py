import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np
import data_prepare
import os

data_pre = data_prepare.Data_Prepare()
parent_path = os.path.dirname(os.getcwd())


class Infer(object):
    """
        ues model to predict classification.
    """
    def __init__(self):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(parent_path+'/save_model' +
                                                                               '/abcnn/vocab.pickle')
        self.checkpoint_file = tf.train.latest_checkpoint(parent_path+'/save_model' + '/abcnn')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.text_a = graph.get_operation_by_name("text_a").outputs[0]
                self.text_b = graph.get_operation_by_name("text_b").outputs[0]

                # Tensors we want to evaluate
                self.prediction = graph.get_operation_by_name("prediction").outputs[0]
                self.score = graph.get_operation_by_name("score").outputs[0]

    def infer(self, sentenceA, sentenceB):
        # transfer to vector
        sentenceA = [data_pre.pre_processing(sentenceA)]
        sentenceB = [data_pre.pre_processing(sentenceB)]
        vector_A = np.array(list(self.vocab_processor.transform(sentenceA)))
        vector_B = np.array(list(self.vocab_processor.transform(sentenceB)))
        feed_dict = {
            self.text_a: vector_A,
            self.text_b: vector_B
        }
        y, s = self.sess.run([self.prediction, self.score], feed_dict)
        return y, s


if __name__ == '__main__':
    infer = Infer()
    sentencea = '你点击详情'
    sentenceb = '您点击详情'
    print(infer.infer(sentencea, sentenceb))
