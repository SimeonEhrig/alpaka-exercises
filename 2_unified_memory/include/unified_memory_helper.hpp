#pragma once

#include "helper.hpp"

// functor for the alpaka::transform
struct Increment
{
    // increment each value
    constexpr auto operator()(alpaka::onAcc::concepts::Acc auto const&, auto const& value) const
    {
        return value + 1;
    }
};

std::tuple<bool, std::string> verify(alpaka::concepts::IDataSource auto& data)
{
    static_assert(data.dim() == 1);
    bool correct = true;

    // build string for error output
    std::stringstream error_string;
    error_string << "Result does not match\nexpected result: [";
    // check if the elements in data matches the expected result
    for(auto i = 0; i < data.getExtents()[0]; ++i)
    {
        auto expected_number = i + 1;
        error_string << expected_number << " ";
        if(data[i] != expected_number)
        {
            correct = false;
        }
    }
    error_string << "]\ndata           : " << dataSourceToStr(data);
    return {correct, error_string.str()};
}
