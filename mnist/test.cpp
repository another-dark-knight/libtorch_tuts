#include <iostream>
#include <torch/torch.h>

int main(){
  torch::Device device= torch::kCPU;
  if(torch::cuda::is_available()){
    std::cout<<"GPU Detected\n";
    device =torch::kCUDA;
  }
  torch::Tensor a = torch::eye(10);
  a = a.to(device);
  std::cout<<a;

  return 0;
}
