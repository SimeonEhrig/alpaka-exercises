

#include "unified_memory_helper.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    // #################################################
    // select device
    // #################################################

    auto device_selector = alpaka::api::cuda + alpaka::deviceKind::nvidiaGpu;

    auto num_devices = device_selector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";

    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device acc_device = device_selector.makeDevice(0);
    std::cout << "Device 0: " << acc_device.getName() << "\n";

    alpaka::onHost::Queue acc_queue = acc_device.makeQueue(alpaka::queueKind::blocking);

    // #################################################
    // allocate memory and apply transformation
    // #################################################

    std::size_t const n = 20;
    auto data = alpaka::onHost::allocUnified<int>(acc_device, n);

    for(auto i = 0; i < 20; ++i)
    {
        data[i] = i;
    }

    // increment each element in data
    alpaka::onHost::transform(acc_queue, data, Increment{}, data);

    // #################################################
    // verify result
    // #################################################

    // check if the result is correct
    auto [correct, error_string] = verify(data);

    if(correct)
    {
        std::cout << "Result is correct\n";
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << error_string << "\n";
        return EXIT_FAILURE;
    }
}
