require_relative "../lib/fast_forward"

filename_prefix = "examples/data/"

json_files_to_check = [
    "mlp_relu_regressor.json",
    # "mlp_logistic_regressor.json",
    # "mlp_tanh_regressor.json",
    # "mlp_relu_classifier.json",
    "mlp_logistic_classifier.json",
    # "mlp_tanh_classifier.json"
]

pmml_files_to_check = [
    "keras_relu_regressor",
    # "keras_sigmoid_regressor",
    # "keras_tanh_regressor",
    # "keras_relu_classifier",
    # "keras_sigmoid_classifier",
    "keras_tanh_classifier"
]


json_files_to_check.each do |filename|
    data = File.read("#{filename_prefix}#{filename}")
    nn = FastForward.load_json(data, tol: 1e-8, exception_if_fail: false)
    puts "#{filename} successfully loaded"
end


pmml_files_to_check.each do |filename|
    nn = FastForward.load_pmml(File.read("#{filename_prefix}#{filename}.pmml"))
    nn.check_model_integrity(FastForward.load_sample_data(File.read("#{filename_prefix}#{filename}.json")), tol: 1e-8)
    puts "#{filename} successfully loaded"
end
