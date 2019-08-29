#include<iostream>
#include<torch/torch.h>

const int8_t epochs = 10;
const int16_t batch_size = 256;
const double learning_rate = 0.001;

struct classifierImpl: torch::nn::Module {
    classifierImpl(){
        layer_1 = register_module("layer_1", torch::nn::Linear(784,196));
        layer_2 = register_module("layer_2", torch::nn::Linear(196, 98));
        layer_3 = register_module("layer_3", torch::nn::Linear(98,49));
        layer_4 = register_module("layer_4", torch::nn::Linear(49,10));
    }
    torch::nn::Linear layer_1{nullptr}, layer_2{nullptr}, layer_3{nullptr}, layer_4{nullptr};

    torch::Tensor forward(torch::Tensor x){
        x = x.reshape({x.size(0), 784});
        x = layer_1->forward(x);
        x = torch::relu(x);
        x = layer_2->forward(x);
        x = torch::relu(x);
        x = torch::dropout(x, 0.5, is_training());
        x = layer_3->forward(x);
        x = torch::relu(x);
        x = torch::dropout(x, 0.2, is_training());
        x = layer_4->forward(x);
        x = torch::log_softmax(x,1);
        return x;
    }
};
TORCH_MODULE(classifier);
int main(){
    classifier net;
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()){
        device = torch::kCUDA;
        std::cout<<"GPU Detected and in use"<<std::endl;
    }
    net->to(device);
    auto train_dataset = torch::data::datasets::MNIST("./data");
    const size_t train_dataset_size = train_dataset.size().value();
    auto test_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest);
    const size_t test_dataset_size = test_dataset.size().value();
    auto train_loader = torch::data::make_data_loader(train_dataset
                                                      .map(torch::data::transforms::Normalize<>(0.1307,0.3081))
                                                      .map(torch::data::transforms::Stack<>()), batch_size);
    auto test_loader = torch::data::make_data_loader(test_dataset
                                                      .map(torch::data::transforms::Normalize<>(0.1307,0.3081))
                                                      .map(torch::data::transforms::Stack<>()), batch_size);
    torch::optim::Adam optimizer(net->parameters(), learning_rate);
    for(size_t epoch = 0; epoch<=epochs; epoch++){
        float eploss=0;
        for(auto& batch: *train_loader){
            net->train();
            optimizer.zero_grad();
            auto x = batch.data.to(device), y =batch.target.to(device);
            auto out = net->forward(x);
            auto loss = torch::nll_loss(out, y);
            loss.backward();
            optimizer.step();
            eploss += loss.item<float>();
        }
        std::cout<<"Epoch : "<<epoch<<"\t||\tLoss: "<<static_cast<float>(eploss)/train_dataset_size<<std::endl;
        float test_loss=0;
        int32_t correct=0;
        for(auto& batch: *test_loader){
            net->eval();
            auto x = batch.data.to(device), y =batch.target.to(device);
            auto out = net->forward(x);
            auto loss = torch::nll_loss(out, y);
            test_loss += loss.item<float>();
            auto pred = out.argmax(1);
            correct += pred.eq(y).sum().template item<int64_t>();
        }
        std::cout<<"Test :: Loss: \t"<<static_cast<float>(test_loss)/test_dataset_size<<"||\t Accuracy : "<<static_cast<float>(correct)/test_dataset_size<<"\n\n";
        torch::save(net,"net.pt");
    }

}
