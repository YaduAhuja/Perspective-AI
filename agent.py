import enum
import numpy as np 
import random
import os
import json
import string
from collections import deque

from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Flatten, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


"""Experience from previous States And Actions"""
class Experience():
	def __init__(self) -> None:
		self.list = []
	
	def add(self, qString, context, attempts, action) -> None:
		self.list.append({"qString":qString, "context":context, "attempts":attempts, "action":action })


"""Agent and the Model"""
class Agent():
	def __init__(self, input_length = 30, context_length = 5) -> None:
		actions = list(string.ascii_lowercase + "=.;_ '()|%")
		# actions = list(string.ascii_lowercase+"%|")
		print("Possible Actions in Action Space : ",actions)

		self.action_depth = len(actions)
		self.action_indices = dict((c,i) for i,c in enumerate(actions))
		self.indices_action = dict((i,c) for i,c in enumerate(actions))

	
		#Building the Model
		
		#Building Context Layers
		context_input = Input(shape=(context_length ,self.action_depth), name="Context_Input")
		context_layer_1 = LSTM(32, name = "Context_Layer_1")(context_input)
		context_layer_2 = Dense(128, name = "Context_Layer_2")(context_layer_1)

		#Building Query Layers
		qString_input = Input(shape=(input_length,self.action_depth), name="QString_Input")
		qString_layer_1 = LSTM(128, name= "QString_Layer_1")(qString_input)
		qString_layer_2 = Dense(128, name="QString_Layer_2")(qString_layer_1)

		#Building Attempts Layers
		attempts_input = Input(shape=(self.action_depth,), name="Attempts_Input")
		attempts_layer_1 = Dense(128, name="Attempts_Layer_1")(attempts_input)

		#Concatenating Layers
		x = keras.layers.concatenate([context_layer_2, qString_layer_2])
		x = Dense(64, name="Hidden_layer_1")(x)
		x = keras.layers.concatenate([x, attempts_layer_1])

		#Stacking the Deep Densly Connected network on top
		main_output = Dense(self.action_depth, activation="softmax", name="Output_Layer")(x)
		self.model = Model(inputs = [context_input, qString_input, attempts_input], outputs = [main_output])
		print(self.model.summary())

		#Compiling the Model
		#Hyperparameters - Need to be Adjusted
		self.supervised_reward = 1
		self.content = None
		self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= ["accuracy"])
		self.temperature = .2
		self.epsilon = .65
		self.input_length = input_length
		self.context_length = context_length


	#Given the current state, choose an action an return it
	#Stochastic! (ie: we choose an action at random, using each state as a probability)
	def act(self, qstring, context, attempts):
		qstring = self.onehot(qstring, self.input_length)
		context = self.onehot(context, self.context_length)
		attempts = self.encodeattempts(attempts)
		#[context_layer, qstring_layer, attempts_layer]
		predictions = self.model.predict_on_batch([context, qstring, attempts])
		action = self.sample(predictions[0], self.temperature)
		#print(predictions, action, self.indices_char[action])
		return self.indices_action[action]

	#Given a stochastic prediction, generate a concrete sample from it
	#temperature: How much we should "smear" the probability distribution. 0 means not at all, high numbers is more.
	def sample(self, preds, temperature=1.0):
		# helper function to sample an index from a probability array
		with np.errstate(divide='ignore'):
			preds = np.asarray(preds).astype('float64')
			preds = np.log(preds) / temperature
			exp_preds = np.exp(preds)
			preds = exp_preds / np.sum(exp_preds)
			probas = np.random.multinomial(1, preds, 1)
			return np.argmax(probas)

	#Update the model for a single given experience
	def train_single(self, action, context, prev_state, attempts, reward):
		#make a one-hot array of our output choices, with the "hot" option
		#   equal to our discounted reward
		action = self.action_indices[action]
		prev_state = self.onehot(prev_state, self.input_length)
		reward_array = np.zeros((1, self.action_depth))
		reward_array[0, action] = reward
		attempts = self.encodeattempts(attempts)
		context = self.onehot(context, self.context_length)
		#[context_layer, qstring_layer, attempts_layer]
		self.model.train_on_batch([context, prev_state, attempts], reward_array)

	# Batch the training
	def train_batch(self, contents, batch_size, epochs, tensorboard=False):
		self.content = contents

		qstrings = np.zeros((len(self.content), self.input_length, self.action_depth))
		contexts = np.zeros((len(self.content), self.context_length, self.action_depth))
		attempts = np.zeros((len(self.content), self.action_depth))
		rewards = np.zeros((len(self.content), self.action_depth))

		for i in range(len(self.content)):
			entry = random.choice(self.content)

			qstrings[i] = self.onehot(entry["qstring"], self.input_length)
			contexts[i] = self.onehot(entry["context"], self.context_length)
			attempts[i] = self.encodeattempts(entry["attempts"])
			reward = self.supervised_reward
			if not entry["success"]:
				reward *= -0.1
			action = self.action_indices[entry["action"]]
			reward_array = np.zeros(self.action_depth)
			reward_array[action] = reward
			rewards[i] = reward_array
		callbacks = []
		if tensorboard:
			callbacks.append(TensorBoard(log_dir='./TensorBoard', histogram_freq=1, write_graph=True))
		self.model.fit([contexts, qstrings, attempts], rewards, callbacks=callbacks,
						epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

	# Given a full experience, go back and reward it appropriately
	def train_experience(self, experience : Experience, success):
		experience.list.reverse()
		reward = self.supervised_reward
		if not success:
			reward *= -0.1
		i = 0
		for item in experience.list:
			i += 1
			if i > 4:
				if item["action"] == "'":
					return
				reward *= self.epsilon
				self.train_single(item["action"], item["context"], item["qstring"], item["attempts"], reward)

	#Encode a given string into a 2d array of one-hot encoded numpy arrays
	def onehot(self, string, length):
		assert len(string) <= length
		#First, pad the string out to be 'input_len' long
		string = string.ljust(length, "|")

		output = np.zeros((1, length, self.action_depth), dtype=np.bool)
		for index, item in enumerate(string):
			output[0, index, self.action_indices[item]] = 1
		return output

	def encodeattempts(self, attempts):
		output = np.zeros((1, self.action_depth))
		for i, item in enumerate(attempts):
			output[0, i] = 1
		return output

	def save(self, path):
		self.model.save_weights(path)
		print("Saved model to disk")

	def load(self, path):
		if os.path.isfile(path):
			self.model.load_weights(path)
			print("Loaded model from disk")
		else:
			print("No model on disk, starting fresh")
