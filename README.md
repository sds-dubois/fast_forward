# FastForward
*A lightweight Ruby gem for fast predictions with neural networks*  


The purpose of this package is to provide a lightweight, pure Ruby (but fast!) implementation to make predictions with neural networks. This means:

- Simple ways to import a neural network trained in another language, from JSON or [PMML](http://dmg.org/pmml/pmml-v4-3.html) files. Note that this package does not support training models.
- Predictions are *fast* since the implementation relies on [NMatrix](https://github.com/SciRuby/nmatrix) and matrix-vector multiplication.


## Installation

Add this line to your application's Gemfile:

```ruby
gem 'fast_forward', :git => "git://github.com/sds-dubois/fast_forward.git"
```

## Usage

[Online documentation](https://sds-dubois.github.io/fast_forward/index.html)

### Load from a JSON file

```ruby
require "fast_forward"

data = File.read("examples/data/mlp_relu_regressor.json")
nn = FastForward.load_json(data)

inputs = [[0.14, 0.1, 1.4, 0.44, 0.76], [0.18, -0.3, 0.52, 0.09, 0.08]]
nn.predict(inputs)
# => [174.49829607413798, 67.8113924823969]
nn.predict(inputs.first)
# => 174.49829607413798
```

### Load from a PMML file

```ruby
require "fast_forward"

filename = "examples/data/keras_tanh_classifier"
data = File.read("#{filename}.pmml")
sample_data = FastForward.load_sample_data(File.read("#{filename}.json"))
nn = FastForward.load_pmml(data)
nn.check_model_integrity(sample_data, tol: 1e-8)
# => true

inputs = [[1.01, 2.35, -2.76, -1.08, -0.46]]
nn.predict_proba(inputs)
# => [0.8352501258487187, 0.13632516324287638, 0.02842471090840499]
nn.predict_class(inputs)
# => 0
nn.predict(inputs)
# => 0
```

It is highly recommended to check model integrity to make sure it correctly matches the
model trained in a different language. One way to do that is by providing sample data (inputs/outputs) from the original model exported, and verify in Ruby that outputs are identical.  
Note that when exporting in PMML format, that means you need to export the sample data in a different file, in JSON. See the [example](examples/export_py_models.py) for more details.


### Export Keras and scikit-learn models

Check the [Python example](examples/export_py_models.py) for ways to export models in a compatible format.

- You can export Keras models with [keras2pmml](https://github.com/sds-dubois/keras2pmml) (currently supports feed-forward models with fully connected layers).  
[Here](examples/data/keras_relu_regressor.pmml) is an example. 
- scikit-learn's `MLP` interface for neural networks is straightforward and easy to export in JSON 
(see the example for detailed code). Be sure to repeat the activation function for each layer and to 
add the output layer activation (linear for regression / sigmoid or softmax for classification).  
[Here](examples/data/mlp_relu_regressor.json) is an example. 

## Support

### Models
- Regression and classification (multi-class and multi-task)
- Feed-forward with fully connected layers
- Activation functions:
    - identity (linear)
    - relu (rectifier)
    - sigmoid (logistic)
    - tanh
    - softmax

### Data import

- Ruby hash

- [PMML](http://dmg.org/pmml/v4-3/NeuralNetwork.html)

- JSON, with the following structure:  

    ```
    data = {
        "weights": {
            "0": weights_array_0,
            "1": weights_array_1,
            ...
        },
        "biases":
            "0": biases_array_0,
            "1": biases_array_1,
            ...
        },
        "activations":
            "0": activation_fct_0,
            "1": activation_fct_1,
            ...
        },
        "samples": (optional) {
            "X": sample_inputs_array,
            "y": sample_outputs_array
        }
    }
    ```

## License

[MIT License](LICENSE.md)
