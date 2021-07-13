import pickle
import keras
import uuid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import msgpack
import random
import codecs
import numpy as np
import json
import msgpack_numpy
# https://github.com/lebedov/msgpack-numpy

import math
import sys
import time
import load_car10

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
from itertools import islice
# https://flask-socketio.readthedocs.io/en/latest/
from src.parsingconfig import readconfig
import ea_datasource
from aggregators import deprecated_native
import experiments
import aggregators
import graph
import cluster
K.set_floatx('float64')

class GlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self):
        self.model, self.graph = self.build_model()
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        self.training_start_time = int(round(time.time()))

        fake_data = ea_datasource.Mnist().fake_non_iid_data('','','')
        #fake_data = load_car10.Car10().fake_non_iid_data('','','')
        train_data, test_data, valid_data = fake_data
        self.global_x_test, self.global_y_test = test_data
        global_x_train, global_y_train = train_data
        global_x_valid, global_y_valid = valid_data

        filename = './log/seq_log'
        f = open(filename,'a')
        f.write('train:test:valid = %s,%s,%s,%s,%s,%s\n\n'%(str(global_x_train.shape),str(global_y_train.shape),str(self.global_x_test.shape),str(self.global_y_test.shape),str(global_x_valid.shape),str(global_y_valid.shape)))
        f.close()

        aggregator = aggregators.instantiate("krum-py", 8, 0, []) #median, krum-py, average

    def eval_global_model(self):
        with self.graph.as_default():
            Model = self.model
            # Model.compile(loss=keras.losses.categorical_crossentropy,
            #     optimizer=keras.optimizers.Adadelta(),
            #     metrics=['accuracy'])
            Model.set_weights(self.current_weights)  
            time.sleep(0.1)   
        with self.graph.as_default():   
            score = Model.evaluate(self.global_x_test, self.global_y_test, verbose=0)
        return score[1]

    def build_model(self):
        raise NotImplementedError()

    def krum_aggregator(self,gradients):
        nbworkers = 5
        nbbyzwrks = 0
        nbselected = nbworkers - nbbyzwrks - 2
        if nbselected == nbworkers:
            # Fast path average
            result = gradients[0].eval()
            for i in range(1, nbworkers):
                result += gradients[i].eval()
            result /= float(nbworkers)
            return result
        else:
            # Compute list of scores
            scores = [list() for i in range(nbworkers)]
            for i in range(nbworkers - 1):
                score = scores[i]
                for j in range(i + 1, nbworkers):
                # With: 0 <= i < j < nbworkers
                    distance = deprecated_native.squared_distance(gradients[i].eval(), gradients[j].eval())
                if math.isnan(distance):
                    distance = math.inf
                score.append(distance)
                scores[j].append(distance)
            nbinscore = nbworkers - nbbyzwrks - 2
            for i in range(nbworkers):
                score = scores[i]
                score.sort()
                scores[i] = sum(score[:nbinscore])
            # Return the average of the m gradients with the smallest score
            pairs = [(gradients[i].eval(), scores[i]) for i in range(nbworkers)]
            pairs.sort(key=lambda pair: pair[1])
            result = pairs[0][0]
            for i in range(1, nbselected):
                result += pairs[i][0]
            result /= float(nbselected)
            return result

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        gradients = []
        temp_gradient = []

        for i in range(len(client_weights)):
            for j in range(len(client_weights[i])):
                client_weights[i][j] = client_weights[i][j].flatten().tolist() 

            client_weights[i] = client_weights[i][0]+client_weights[i][1]+client_weights[i][2]+client_weights[i][3]+client_weights[i][4]+client_weights[i][5]
            gradient = tf.convert_to_tensor(client_weights[i])
            gradients.append(gradient)
                    
        sess = tf.Session()
        with sess.as_default():
        
            ####### Median #######
            # gradients = tf.parallel_stack(gradients)
            # new_weights = deprecated_native.median(gradients.eval())

            ####### Krum #########
            new_weights = self.krum_aggregator(gradients)

            ####### Average ######
            # new_weights = (tf.add_n(gradients) / float(len(gradients))).eval()      #new_weights:<class 'numpy.ndarray'>

        weights = [list(islice(iter(new_weights), i)) for i in [288,32,18432,64,92160,10]]
        #weights = [list(islice(iter(new_weights), i)) for i in [864,32,18432,64,125440,10]]
        for i in range(6):
            if i == 0:
                weights[i] = np.array(weights[i]).reshape(3,3,1,32)
                #weights[i] = np.array(weights[i]).reshape(3,3,3,32)
            elif i == 2:
                weights[i] = np.array(weights[i]).reshape(3,3,32,64)
            elif i == 4:
                weights[i] = np.array(weights[i]).reshape(9216,10)
                #weights[i] = np.array(weights[i]).reshape(12544,10)
            else:
                weights[i] = np.array(weights[i])       
        self.current_weights = weights

        # new_weights = [np.zeros(w.shape) for w in self.current_weights] # has 6 features
        # total_size = np.sum(client_sizes)

        # for c in range(len(client_weights)):
        #     for i in range(len(new_weights)):
        #         new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size # each weight(3x3x32x1) * 800 / 5*800
        # self.current_weights = new_weights 

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        # cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.train_losses += [[cur_round, cur_time, aggr_loss]]
        # self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        # with open('stats.txt', 'w') as outfile:
        #     json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        # cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        # self.valid_losses += [[cur_round, cur_time, aggr_loss]]
        # self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        # with open('stats.txt', 'w') as outfile:
        #     json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # def get_stats(self):
    #     return {
    #         "train_loss": self.train_losses,
    #         "valid_loss": self.valid_losses,
    #         "train_accuracy": self.train_accuracies,
    #         "valid_accuracy": self.valid_accuracies
    #     }
        

class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    # def build_model(self):
    #     # ~5MB worth of parameters
    #     model = Sequential()
    #     model.add(Conv2D(32, kernel_size=(3, 3),
    #                      activation='relu',
    #                      input_shape=(28, 28, 1)))
    #     model.add(Conv2D(64, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))
    #     model.add(Flatten())
    #     # model.add(Dense(128, activation='relu'))
    #     # model.add(Dropout(0.5))
    #     model.add(Dense(10, activation='softmax'))

    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #                   optimizer=keras.optimizers.Adadelta(),
    #                   metrics=['accuracy'])
    #     #K.clear_session()
    #     print(model.summary())
    #     return model, tf.get_default_graph()

    def build_model(self):
        # ~5MB worth of parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         #input_shape=(32, 32, 3)))
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        print(model.summary())    # (Mnist)Total params: 110,986 and 1,199,882; (Car10)Total params: 144,842 and 1,626,442
        return model, tf.get_default_graph()

    # def build_model(self):
    #     model = Sequential([
    #     Dense(64, activation='relu', input_shape=(784,)),
    #     Dense(64, activation='relu'),
    #     Dense(10, activation='softmax'),
    #     ])
    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #                   optimizer=keras.optimizers.Adadelta(),
    #                   metrics=['accuracy'])
    #     print(model.summary())    # (Mnist)Total params: 110,986 and 1,199,882; (Car10)Total params: 144,842 and 1,626,442
    #     return model, tf.get_default_graph()

        
######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):
    
    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.converges_review = []
        self.agg_train_accuracy = []
        self.aggr_valid_accuracy = []
        #####

        # socket io messages
        self.register_handles()

        self.connect_start = []
        self.connect_end = []
        self.request_train_start = []
        self.request_train_end = []
        self.agg_start = []
        self.agg_end = []
        self.computation_start = []
        self.computation_end = []


        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

        
    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")
            self.connect_start.append(time.time())

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            self.connect_end.append(time.time())
            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    #'model_id': self.model_id,
                    'min_train_size': 800,###200
                    'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'epoch_per_round': 1,
                    'batch_size': 10
                })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid, data)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", request.sid)
            # for x in data:
            #     if x != 'weights':
            #         print(x, data[x])
            # data:
            #   weights
            #   train_size
            #   valid_size
            #   train_loss
            #   train_accuracy
            #   valid_loss?
            #   valid_accuracy?
            #self.agg_start.append(time.time())
            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                
                # tolerate 30% unresponsive clients
                if len(self.current_round_client_updates) >= int(NUM_CLIENTS_CONTACTED_PER_ROUND):
                    self.computation_start.append(time.time())
                    self.agg_start.append(time.time())
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    self.agg_end.append(time.time())
                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    self.computation_end.append(time.time())
                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)

                    # if 'valid_loss' in self.current_round_client_updates[0]:
                    #     aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                    #         [x['valid_loss'] for x in self.current_round_client_updates],
                    #         [x['valid_accuracy'] for x in self.current_round_client_updates],
                    #         [x['valid_size'] for x in self.current_round_client_updates],
                    #         self.current_round
                    #     )
                    #     print("aggr_valid_loss", aggr_valid_loss)
                    #     print("aggr_valid_accuracy", aggr_valid_accuracy)

                    global_valid_accuracy = self.global_model.eval_global_model()
                    
                    try:
                        self.converges_review.append((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss)
                        self.agg_train_accuracy.append(aggr_train_accuracy)
                        self.aggr_valid_accuracy.append(global_valid_accuracy)
                    except:
                        pass

                    #print("Current aggregate train accuracy: ",self.agg_train_accuracy)
                    self.save_accuracy_to_file()

                    # if self.global_model.prev_train_loss is not None and \
                    #         abs((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss) < .01 and \
                    #         abs((self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss) != 0.0:
                    #     # converges
                    #     print("converges! starting test phase..")
                    #     self.stop_and_eval()
                    #     return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round > MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) >= int(NUM_CLIENTS_CONTACTED_PER_ROUND):
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                )
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again

                #print((self.connect_end[0]-self.connect_start[0])+(self.request_train_end[0]-self.request_train_start[0]))
                # for i in range(len(self.request_train_end)):
                #     latency_per_epoch = (self.request_train_end[i]-self.request_train_start[i])+(self.computation_end[i]-self.computation_start[i])
                #     print("latency_per_epoch", latency_per_epoch*1000)
                for i in range(len(self.agg_end)):
                    aggr_time = self.agg_end[i]-self.agg_start[i]
                    print("aggr time", aggr_time*1000)

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self):
        self.request_train_start.append(time.time())
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), int(NUM_CLIENTS_CONTACTED_PER_ROUND))
        print("request updates from", client_sids_selected)
        self.request_train_end.append(time.time())
        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            emit('request_update', {
                    #'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    #'weights_format': 'pickle',
                    #'run_validation': self.current_round % ROUNDS_BETWEEN_VALIDATIONS == 0,
                    #'run_validation': True,
                }, room=rid)

    
    def stop_and_eval(self):
        self.eval_client_updates = []
        #print("converges review list: ",self.converges_review)
        #print("aggregate train accuracy list: ",self.agg_train_accuracy)
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                    #'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    #'weights_format': 'pickle'
                }, room=rid)

    def save_accuracy_to_file(self):
        print("Current converge status: ",self.converges_review)
        print("Current aggregate accuracy change: ",self.agg_train_accuracy)
        print("[Evaluation] Global model validate accuracy change: ",self.aggr_valid_accuracy)
        
        # filename = './log/seq_log'
        # f = open(filename,'a')
        # f.write('Round: '+str(self.current_round)+"\nCurrent aggregate accuracy change: "+str(self.agg_train_accuracy)+"\n[Evaluation] Global model validate accuracy change: "+str(self.aggr_valid_accuracy)+'\n\n')
        # f.close()

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    nodes, servers, clients, baseport, LOCAL, ip_address = readconfig(0)

    MIN_NUM_WORKERS = int(nodes)
    MAX_NUM_ROUNDS = 30
    NUM_CLIENTS_CONTACTED_PER_ROUND = MIN_NUM_WORKERS
    ROUNDS_BETWEEN_VALIDATIONS = 5

    if LOCAL == 1:
        server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
        print("listening on 127.0.0.1:5000")
        server.start()
    elif LOCAL == 0:
        server = FLServer(GlobalModel_MNIST_CNN, str(ip_address), int(baseport))
        print("listening on %s:%s"%(str(ip_address),str(baseport)))
        server.start()
