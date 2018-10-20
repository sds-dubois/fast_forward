# dependencies
require "nmatrix"
require "nokogiri"
require "json"

# module
require_relative "fast_forward/version"
require_relative "fast_forward/nn"


module FastForward
	# Default tolerance for integrity check
	DEFAULT_TOL = 1e-8

	# Supported activation functions
	SUPPORTED_ACTIVATIONS = ["identity", "relu", "sigmoid", "tanh", "softmax"].freeze


	# Loads neural network data into a NN object
	# @param data [Hash] A hash containing all weights, biases and activation functions
	# @param tol [Numeric] (Default: 1e-8) The tolerance to check integrity of the imported model from the provided samples
	# @param exception_if_fail [true, false] (Default: true) If true, raises an exception if integrity check doesn't meet the required tolerance
	# @return The loaded neural network
	# @note data is supposed to have the following structure:
	# 	data = {
	# 		weights: [weights_array_0, weights_array_1 , ...],
	# 		biases: [biases_array_0, biases_array_1 , ...],
	# 		activations: [activation_fct_0, activation_fct_1, ...],
	# 		samples: (optional) {
	#			X: sample_inputs_array,
	#			y: sample_outputs_array
	# 		}
	# 	}

	def self.load(data, tol: DEFAULT_TOL, exception_if_fail: true)

		# check inputs
		if data[:weights].nil? || data[:weights].count == 0
			raise ArgumentError, "Missing weights data"
		elsif data[:biases].nil? || data[:biases].count == 0
			raise ArgumentError, "Missing biases data"
		elsif data[:activations].nil? || data[:activations].count == 0
			raise ArgumentError, "Missing activation functions"
		elsif data[:weights].size != data[:biases].size
			raise ArgumentError, "Array of weights and biases must have the same size"
		elsif data[:weights].size != data[:activations].size
			raise ArgumentError, "Array of weights and activations must have the same size"
		elsif !data[:activations].map{ |act| SUPPORTED_ACTIVATIONS.include?(act) }.all?
			raise ArgumentError, "Some activation functions are not supported"
		end


		layer_sizes = data[:weights].map(&:count) + [data[:weights].last.first.count]

		nn = NN.new(layer_sizes, data[:activations], data[:weights], data[:biases])

		if !data[:samples].nil? && tol >= 0
			FastForward.check_model_integrity(nn, data[:samples], tol: tol, exception_if_fail: exception_if_fail, verbose: true)
		end

		return nn
	end



	# Loads neural network from JSON data/string into a NN object
	# @param data [Hash, String] A JSON string/object containing all weights, biases and activation functions
	# @param tol [Numeric] (Default: 1e-8) The tolerance to check integrity of the imported model from the provided samples
	# @param exception_if_fail [true, false] (Default: true) If true, raises an exception if integrity check doesn't meet the required tolerance
	# @return The loaded neural network
	# @note data is supposed to have the following structure:
	# 	data = {
	# 		"weights": {
	# 			"0": weights_array_0,
	# 			"1": weights_array_1,
	# 			...
	# 		},
	# 		"biases":
	# 			"0": biases_array_0,
	# 			"1": biases_array_1,
	# 			...
	# 		},
	# 		"activations":
	# 			"0": activation_fct_0,
	# 			"1": activation_fct_1,
	# 			...
	# 		},
	# 		"samples": (optional) {
	#			"X": sample_inputs_array,
	#			"y": sample_outputs_array
	# 		}
	# 	}

	def self.load_json(data, tol: DEFAULT_TOL, exception_if_fail: true)

		data = JSON.parse(data) if data.is_a?(String)

		if data["weights"].keys != data["biases"].keys
			raise ArgumentError, "Incoherent weights and biases data"
		end
		n_layers = data["weights"].count

		# data["weights"] & data["biases"] are dictionaries of `"idx": array`
		d = {
			weights: n_layers.times.map{ |i| data["weights"][i.to_s] },
			biases: n_layers.times.map{ |i| data["biases"][i.to_s] },
			activations: n_layers.times.map{ |i| FastForward.rename_activation(data["activations"][i.to_s]) },
			samples: FastForward.load_sample_data(data)
		}

		return FastForward.load(d, tol: tol, exception_if_fail: exception_if_fail)
	end


	
	# Loads neural network from PMML file into a NN object
	# @param data [String] PMML file content
	# @param sample_data [Hash, String] (Default: nil) A JSON string/object containing sample data to check the integrity of the loaded model
	# @param tol [Numeric] (Default: 1e-8) The tolerance to check integrity of the imported model from the provided samples
	# @param exception_if_fail [true, false] (Default: true) If true, raises an exception if integrity check doesn't meet the required tolerance
	# @return The loaded neural network
	# @note sample_data is supposed to have the following structure:
	# 	sample_data = {
	# 		"samples": {
	#			"X": sample_inputs_array,
	#			"y": sample_outputs_array
	# 		}
	# 	}

	def self.load_pmml(data, sample_data: nil, tol: DEFAULT_TOL, exception_if_fail: true)
		# Load from PMML format: http://dmg.org/pmml/v4-3/NeuralNetwork.html
		# Note: does not support input transforms like NormContinuous
		# TODO: use element ids to check order
		data = Nokogiri::XML(data) if data.is_a?(String)
		
		all_weights = []
		all_biases = []
		all_activations = []
		
		default_act = data.at_css("NeuralNetwork").attribute("activationFunction").value
		
		layers = data.css("NeuralLayer")
		layers.each do |layer|
			w = []
			b = []
			layer.css("Neuron").each do |neuron|
				b << neuron.attribute("bias").value.to_f
				w << neuron.css("Con").map{ |wi| wi.attribute("weight").value.to_f }
			end
			all_weights << w.transpose
			all_biases << b

			# in PMML format, softmax is a normalization and not an activation
			normalization = layer.attribute("normalizationMethod") && layer.attribute("normalizationMethod").value
			if normalization == "softmax"
				activation = "softmax" 
			else
				layer_activation = layer.attribute("activationFunction")
				activation = layer_activation.nil? ? default_act : layer_activation.value
			end

			all_activations << FastForward.rename_activation(activation)
		end

		d = {
			weights: all_weights,
			biases: all_biases,
			activations: all_activations
		}
		d[:samples] = FastForward.load_sample_data(sample_data) unless sample_data.nil?

		return FastForward.load(d, tol: tol, exception_if_fail: exception_if_fail)

	end


	
	# Loads sample data
	# @param data [Hash, String] A JSON string/object containing sample data to check the integrity of the loaded model
	# @return The loaded data in a ruby hash
	# @note data is supposed to have the following structure:
	# 	data = {
	# 		"samples": {
	#			"X": sample_inputs_array,
	#			"y": sample_outputs_array
	# 		}
	# 	}

	def self.load_sample_data(data)
		data = JSON.parse(data) if data.is_a?(String)
		d = {
			X: data["samples"]["X"],
			y: data["samples"]["y"]
		}
		return d
	end


	private

		def self.check_model_integrity(nn, sample_data, tol: DEFAULT_TOL, exception_if_fail: true, verbose: true)
			sample_preds = nn.forward_pass(sample_data[:X])
			original_preds = sample_data[:y]

			if nn.output_dim > 1
				# flatten predictions if more than one output per input
				sample_preds = sample_preds.reduce([], &:+)
				original_preds = original_preds.reduce([], &:+)
			end

			errors = sample_preds.zip(original_preds).map{ |y1,y2| (y1 - y2).abs }

			if errors.max > tol
				msg = "Integrity check failed, with max error: #{errors.max}"
				if exception_if_fail
					raise RuntimeError, msg
				else
					puts "[WARNING] #{msg}"
				end
				return false

			else
				puts "Checked model integrity with sample data and #{tol} tolerance" if verbose
				return true

			end

		end

		def self.rename_activation(activation_fct)
			if activation_fct == "rectifier"
				return "relu"
			elsif activation_fct == "logistic"
				return "sigmoid"
			elsif activation_fct == "linear"
				return "identity"
			else
				return activation_fct
			end
		end

end
