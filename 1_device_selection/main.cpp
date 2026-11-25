#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    // specify a device selector for the alpaka::api::cuda and alpaka::deviceKind::nvidiaGpu
    auto device_selector = /* TODO: create device selector */;

    auto num_devices = device_selector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";

    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device acc_device = /* TODO: select device 0 */;
    // should display something like: Nvidia ...
    std::cout << "Device 0: " << acc_device.getName() << "\n";

    alpaka::onHost::Queue acc_queue = /* TODO: create a blocking queue */;

    return 0;
}
