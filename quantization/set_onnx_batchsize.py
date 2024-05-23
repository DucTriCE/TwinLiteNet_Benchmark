import onnx
import sys

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
#    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
#dim1.dim_param = sym_batch_dim
        # or update it to be an actual value:
        dim1.dim_value = actual_batch_dim


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)

apply(change_input_dim, sys.argv[1], sys.argv[2])
'''

model = onnx.load(sys.argv[1])
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '4'
onnx.save(model, sys.argv[2])
onnx.checker.check_model(model)
'''
