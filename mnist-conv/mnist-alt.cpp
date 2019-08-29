#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>

const int8_t epochs = 20;
const int16_t batch_size = 256;
const double learning_rate = 0.001;

struct classifer : torch::nn::Module {

    classifer() :   conv_l1(torch::nn::Conv2dOptions(1, 10, 5)),
        conv_l2(torch::nn::Conv2dOptions(10, 20, 5)),
        linear_l3(320, 50),
        linear_l4(50, 10)
    {
        register_module("conv_l1", conv_l1);
        register_module("conv_l2", conv_l2);
        register_module("conv_drop", conv_drop);
        register_module("linear_l3", linear_l3);
        register_module("linear_l4", linear_l4);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv_l1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv_drop->forward(conv_l2->forward(x)), 2));
        x = x.view({-1, 320});
        x = linear_l4->forward(torch::dropout(torch::relu(linear_l3->forward(x)),0.5, is_training()));
        x = torch::log_softmax(x, 1);
        return x;
    }
    torch::nn::Conv2d conv_l1, conv_l2;
    torch::nn::FeatureDropout conv_drop;
    torch::nn::Linear linear_l3, linear_l4;
};

template <typename DataLoader>
void train(size_t epoch, classifer& model, torch::Device device, DataLoader& data_loader, torch::optim::Optimizer& optimizer,size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<double>())) //Make sure loss is not NAN
    loss.backward();
    optimizer.step();
    if (batch_idx++ % 10 == 0) {
      std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f", epoch, batch_idx * batch.data.size(0),dataset_size,loss.template item<float>());
    }
  }
}

template <typename DataLoader>
void test(classifer& model, torch::Device device, DataLoader& data_loader, size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model.eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model.forward(data);
    test_loss += torch::nll_loss(output, targets, {},Reduction::Sum).template item<double>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }
  test_loss /= dataset_size;
  std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n",test_loss,static_cast<double>(correct) / dataset_size);
}

auto main() -> int {
  torch::manual_seed(1);
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  classifer model;
  model.to(device);
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
  torch::optim::Adam optimizer(model.parameters(), learning_rate);
  for (size_t epoch = 1; epoch <= epochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);
//    std::printf("\n\n");
  }
}
