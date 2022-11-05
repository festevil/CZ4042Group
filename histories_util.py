import json
import matplotlib.pyplot as plt
import tensorflow.keras as keras
#Code referenced from https://github.com/tanchuanxin/cz4042_assignment_1/blob/main/1a/assignment_1a_q1.ipynb
from time import time



def histories_saver(histories, filename, already_json=False):
    histories_json = {}
    
    if already_json:
        histories_json = histories
    else:
        for key in histories.keys():
            histories_json[key] = histories[key].history

    with open(filename, 'w') as file:
        json.dump(histories_json, file)

    print("Histories saved")

def time_saver(timedict, filename):
    with open(filename, 'w') as file:
        json.dump(timedict, file)

# filename like 'data/q0_histories.json'
def histories_loader(filename):
    with open(filename) as json_file:
        histories = json.load(json_file)
    print('Histories loaded')
    
    return histories 



def plot_history_object(histories, model_name, history_object, val = False):    
    plt.plot(histories[model_name][history_object], label='train_'+history_object)
    
    if val:
        plt.plot(histories[model_name]['val_'+history_object], label='val_'+history_object)

    plt.title(history_object + ' against epochs')
    plt.ylabel(history_object)
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('images/' + model_name + history_object + '.png')
    plt.show()

def plot_history_object_test(histories, model_name, history_object):    
    plt.plot(histories[model_name][history_object], label='train_'+history_object)
    
    plt.title(history_object + ' against epochs')
    plt.ylabel(history_object)
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig('images/' + model_name + history_object + '.png')
    plt.show()

def plot_avg_validation_acc(batch_dict):
    plt.plot(batch_dict['batch_sizes'], batch_dict['avg_val_acc'], label='val_acc')
    plt.title('Average Cross Validation Accuracy against Batch Size')
    plt.ylabel('avg_val_acc')
    plt.xlabel('batch_sizes')
    plt.legend()
    plt.savefig('images/batch_vs_val_acc.png')
    plt.show()

def plot_avg_validation_acc_num_neurons(neurons_dict):
    plt.plot(neurons_dict['neuron_nums'], neurons_dict['avg_val_acc'], label='val_acc')
    plt.title('Average Cross Validation Accuracy against Number of neurons')
    plt.ylabel('avg_val_acc')
    plt.xlabel('neuron_nums')
    plt.legend()
    plt.savefig('images/neurons_vs_val_acc.png')
    plt.show()

def plot_avg_time_taken(batch_dict):
    plt.plot(batch_dict['batch_sizes'], batch_dict['avg_time_taken'], label ='time_taken')
    plt.title('Average Time against Batch Size')
    plt.ylabel('avg_time_taken')
    plt.xlabel('batch_sizes')
    plt.legend()
    plt.savefig('images/batch_vs_time.png')
    plt.show()

def train_set_histories_objects_save(histories_test, histories_model, histories_model_filename):
    histories_saver(histories_model, histories_model_filename)
    histories_model = histories_loader(histories_model_filename)

    histories_model['test_values'] = histories_test['test_values']

    histories_saver(histories_model, histories_model_filename, already_json=True)
    histories_model = histories_loader(histories_model_filename)
    
    return histories_model

def train_set_histories_objects(): 
    # the history object to keep performance of X_test Y_test from final model
    histories_test = {
        'test_values': {
            'loss': [],
            'accuracy': [],
            
        }
    }

    # the history object to keep performance of X_train Y_train from final model
    histories_model = {}
    
    return histories_test, histories_model

# custom callback to evaluate the test set at each epoch
class TestCallback(keras.callbacks.Callback):
    def __init__(self, X_test, Y_test, model_name, histories_test):
        self.X_test = X_test
        self.Y_test = Y_test
        self.histories_test = histories_test
        self.model_name = model_name
        self.histories_test[self.model_name] = {
            'loss': [],
            'accuracy': [],
            
        }

    def on_epoch_end(self, epoch, logs={}):
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        
        
        self.histories_test[self.model_name]['accuracy'].append(accuracy)
        self.histories_test[self.model_name]['loss'].append(loss)


# custom callback to evaluate the test set at each epoch
class TestCallbackReg(keras.callbacks.Callback):
    def __init__(self, test_ds, model_name, histories_test):
        self.test_ds = test_ds
        self.histories_test = histories_test
        self.model_name = model_name
        self.histories_test[self.model_name] = {
            'loss': [],
            'r2': [],
            'rmse':[]
        }

    def on_epoch_end(self, epoch, logs={}):
        loss, r2, rmse = self.model.evaluate(self.test_ds, verbose=0)   
        
        self.histories_test[self.model_name]['loss'].append(loss)
        self.histories_test[self.model_name]['r2'].append(r2)
        self.histories_test[self.model_name]['rmse'].append(rmse)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, model_name, timedict):
        self.timedict = timedict
        self.model_name = model_name
        self.timedict[self.model_name] = []
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()
    def on_epoch_end(self, epoch, logs={}):
        self.timedict[self.model_name].append(time()-self.starttime)

