module FastForward

    class NN
        
        attr_reader :layer_sizes, :layer_activations, :input_dim, :output_dim, :weights, :biases

        def initialize(layer_sizes, layer_activations, weights, biases)
            # layer_sizes must include input_dim and output_dim
            @input_dim = layer_sizes.first
            @output_dim = layer_sizes.last
            @layer_sizes = layer_sizes
            @layer_activations = layer_activations            

            @n_layers = @layer_sizes.count - 1
            @weights = []
            @biases = []
            @n_layers.times.map do |idx|
                shape = [@layer_sizes[idx], @layer_sizes[idx + 1]]
                @weights << NMatrix.new(shape, weights[idx].flatten, dtype: :float64)
                @biases << NMatrix.new([1, shape[1]], biases[idx], dtype: :float64)
            end
        end


        # Check model integrity against sample data
        # @param sample_data [Hash, String] (Default: nil) A JSON string/object containing sample data to check the integrity of the loaded model
	    # @param tol [Numeric] (Default: 1e-8) The tolerance to check integrity of the imported model from the provided samples
        # @param exception_if_fail [true, false] (Default: false) If true, raises an exception if integrity check doesn't meet the required tolerance
        # @param verbose [true, false] (Default: false)
        # @note sample_data is supposed to have the following structure:
        # 	sample_data = {
        # 		"samples": {
        #			"X": sample_inputs_array,
        #			"y": sample_outputs_array
        # 		}
        # 	}
        # @return [true, false] The check verdict

        def check_model_integrity(sample_data, tol: FastForward::DEFAULT_TOL, exception_if_fail: false, verbose: false)
            return FastForward.check_model_integrity(self, sample_data, tol: tol, exception_if_fail: exception_if_fail, verbose: verbose)
        end


        # Neural network forward pass
        # @param inputs [Array] Array of input data
        # @param array_output [true, false] (Default: true)
        #   If true, convert the output to a numerical array.
        #   Otherwise the output will be an NMatrix object
        # @return Network predictions
        
        def forward_pass(inputs, array_output = true)
            # fix input shape if only one element is provided
            inputs = [inputs] if inputs.first.is_a?(Numeric)

            ones = NVector.ones(inputs.count)
            x = NMatrix.new([inputs.count, @input_dim], inputs.flatten, dtype: :float64)
            @n_layers.times.map do |idx|
                h = x.dot(@weights[idx]) + ones.dot(@biases[idx])
                x = NN.activate(h, @layer_activations[idx])
            end

            if !array_output
                return x
            # elsif x.shape.first == 1
            #     return x.first
            elsif x.shape == [1, 1]
                return x.first
            elsif x.shape.last == 1
                return x.to_a.map(&:first)
            else
                return x.to_a
            end
        end


        
        # Predict class probabilities with a forward pass
        # @param inputs [Array] Array of input data
        # @param array_output [true, false] (Default: true)
        #   If true, convert the output to a numerical array.
        #   Otherwise the output will be an NMatrix object
        # @return Network predictions
        # @note Same as forward_pass

        def predict_proba(inputs, array_output = true)
            return forward_pass(inputs, array_output)
        end


        # Predict class labels with a forward pass (supports multi-class and multi-task).
        # @param inputs [Array] Array of input data
        # @return Network predictions

        def predict_class(inputs)
            n_inputs = inputs.first.is_a?(Numeric) ? 1 : inputs.count
            probas = forward_pass(inputs, array_output = true)
            probas = [probas] if n_inputs == 1
            last_act = FastForward.rename_activation(@layer_activations.last)

            if last_act == "softmax"
                # multiclass
                classes = probas.map do |class_p|
                    _,idx = class_p.each_with_index.max
                    idx
                end
            
            elsif probas.first.is_a?(Numeric)
                # 2-class
                classes = probas.map(&:round)

            else
                # multi-task
                classes = probas.map do |task_p|
                    task_p.map(&:round)
                end

            end

            classes = classes.first if n_inputs == 1

            return classes
        end

        
        # Predict outputs with a forward pass.
        # @param inputs [Array] Array of input data
        # @param array_output [true, false] (Default: true)
        #   If true, convert the output to a numerical array.
        #   Otherwise the output will be an NMatrix object
        # @return Network predictions. 
        #   For classification, this will return class labels (otherwise use predict_proba).

        def predict(inputs, array_output = true)
            last_act = FastForward.rename_activation(@layer_activations.last)
            if last_act == "softmax" || last_act == "sigmoid"
                return predict_class(inputs)
            else
                return forward_pass(inputs, array_output)
            end
        end


        # Activation functions
        def self.activate(x, activation_fct)
            if activation_fct == "identity" || activation_fct == "linear"
                return x
            elsif activation_fct == "relu" || activation_fct == "rectifier"
                return NN.relu(x)
            elsif activation_fct == "sigmoid" || activation_fct == "logistic"
                return NN.sigmoid(x)
            elsif activation_fct == "tanh"
                return NN.tanh(x)
            elsif activation_fct == "softmax"
                return NN.softmax(x)
            else
                raise ArgumentError, "Activation function not supported: #{activation_fct}"
            end
        end


        # ReLU activation function
        def self.relu(x)
            x.map!{ |e| [0, e].max }
            return x 
        end
        
        # Sigmoid / logistic activation function
        def self.sigmoid(x)
            x.map!{ |e| 1.0 / (1.0 + Math::exp(-e)) }
            return x
        end
        
        # Tanh activation function
        def self.tanh(x)
            x.map!{ |e| Math::tanh(e) }
            return x
        end
        
        # Softmax activation function
        def self.softmax(x, subtract_max = true)
            if subtract_max
                # better for numerical stability
                # matrix with same shape as x, with max entry per row
                row_max = NN.extend_vec(x.max(1), x.cols)
                x = x - row_max
            end

            e_x = x.exp
            row_sum = NN.extend_vec(e_x.sum(1), e_x.cols)

            return e_x / row_sum
        end


        private

            def self.extend_vec(x, n_cols)
                # [[x1], [x2], ...] => [[x1 x1 x1], [x2 x2 x2], ...]
                return x.dot(NMatrix.ones([1, n_cols]))
            end

    end

end