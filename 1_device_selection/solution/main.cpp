#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    // specify a device selector for the alpaka::api::cuda and alpaka::deviceKind::nvidiaGpu
    auto device_selector = alpaka::api::cuda + alpaka::deviceKind::nvidiaGpu;

    auto num_devices = device_selector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";

    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device acc_device = device_selector.makeDevice(0);
    // should display something like: Nvidia ...
    std::cout << "Device 0: " << acc_device.getName() << "\n";

    alpaka::onHost::Queue acc_queue = acc_device.makeQueue(alpaka::queueKind::blocking);

    return 0;
}
