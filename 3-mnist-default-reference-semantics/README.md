### Using Reference Semantics as default 

In tutorial 1, we used std::make_shared<> to have a reference to the model.
In tutorial 2, we used the model itself.
The python API uses References only and we do not have to explicitly ask it to be so.
This can be achieved by the C++ API as shown here.

A macro, TORCH_MODULE then defines the actual class.

Note the difference of this approach with tutorial 1

![Training](/3-mnist-default-reference-semantics/mnist-alt.png)
