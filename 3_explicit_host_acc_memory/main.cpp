#include <alpaka/alpaka.hpp>

#include <explicit_memory_helper.hpp>

#include <iostream>
#include <sstream>
#include <string>

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
    // allocate memory on host
    auto host_data = /* TODO: allocate memory on the host with size n */;

    for(auto i = 0; i < 20; ++i)
    {
        host_data[i] = i;
    }

    // allocate memory on accelerator
    auto acc_data = /* TODO: allocate memory on the device with size n */;

    /* TODO: copy the memory from the host to the device */

    alpaka::onHost::transform(acc_queue, acc_data, Increment{}, acc_data);

    /* TODO: copy the memory from the device to the host */

    // #################################################
    // verify result
    // #################################################

    // check if the result is correct
    auto [correct, error_string] = verify(host_data);

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

    return 0;
}
