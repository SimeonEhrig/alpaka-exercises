#include "vector_add_helper.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <sstream>
#include <string>

struct Kernel
{
    ALPAKA_FN_ACC void operator()(
        alpaka::onAcc::concepts::Acc auto const& acc,
        alpaka::concepts::IMdSpan auto c,
        alpaka::concepts::IMdSpan auto const a,
        alpaka::concepts::IMdSpan auto const b) const
    {
        for(auto idx :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{c.getExtents()}))
        {
            c[idx] = a[idx] + b[idx];
        }
    }
};

int main()
{
    // #################################################
    // select device
    // #################################################

    auto device_selector = alpaka::api::cuda + alpaka::deviceKind::nvidiaGpu;;

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
    // allocate, initialize and copy the memory
    // #################################################

    std::size_t const n = 20;
    auto host_a = alpaka::onHost::allocHost<int>(n);
    auto host_b = alpaka::onHost::allocHost<int>(n);

    for(auto i = 0; i < 20; ++i)
    {
        host_a[i] = i;
        host_b[i] = 4;
    }

    auto acc_a = alpaka::onHost::allocLike(acc_device, host_a);
    auto acc_b = alpaka::onHost::allocLike(acc_device, host_b);
    auto acc_c = alpaka::onHost::allocLike(acc_device, host_b);

    alpaka::onHost::memcpy(acc_queue, acc_a, host_a);
    alpaka::onHost::memcpy(acc_queue, acc_b, host_b);

    // #################################################
    // launch the kernel
    // #################################################

    Kernel kernel{};
    auto frameSpec = alpaka::onHost::getFrameSpec(acc_device, alpaka::exec::anyExecutor, acc_c.getExtents());
    auto kernel_bundle = alpaka::KernelBundle{kernel, acc_c, acc_a, acc_b};
    acc_queue.enqueue(frameSpec, kernel_bundle);

    auto host_c = alpaka::onHost::allocHost<int>(n);
    alpaka::onHost::memcpy(acc_queue, host_c, acc_c);

    // #################################################
    // verify result
    // #################################################

    // check if the result is correct
    auto [correct, error_string] = verify(host_c);

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
